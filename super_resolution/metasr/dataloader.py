import sys
import threading
#import queue
import random
#import collections

import torch
import torch.multiprocessing as multiprocessing

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import BatchSampler

from torch.utils.data import _utils
from torch.utils.data._utils import collate
from torch.utils.data._utils import signal_handling
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils import ExceptionWrapper
from torch.utils.data._utils import IS_WINDOWS
from torch.utils.data._utils.worker import ManagerWatchdog

#from torch._six import queue
import queue

def _ms_loop(dataset, index_queue, data_queue, done_event, collate_fn, scale, scale2, seed, init_fn, worker_id):
    try:
        collate._use_shared_memory = True
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                assert done_event.is_set()
                return
            elif done_event.is_set():
                continue

            idx, batch_indices = r

            try:
                idx_scale = 0
                if len(scale) * len(scale2) > 1 and dataset.train:
                    idx_scale = random.randrange(0, len(scale)*len(scale2))
                    dataset.set_scale(idx_scale)

                samples = collate_fn([dataset[i] for i in batch_indices])
                samples.append(idx_scale)

            except Exception:
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples
    except KeyboardInterrupt:
        pass

class _MSDataLoaderIter(_DataLoaderIter):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.scale2 = loader.scale2
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [
                multiprocessing.Queue() for _ in range(self.num_workers)
            ]
            self.worker_queue_idx = 0
            #self.worker_result_queue = multiprocessing.SimpleQueue()
            self.worker_result_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = multiprocessing.Event()

            base_seed = torch.LongTensor(1).random_()[0]

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()
                w = multiprocessing.Process(
                    target = _ms_loop,
                    args=(
                        self.dataset,
                        index_queue,
                        self.worker_result_queue,
                        self.done_event,
                        self.collate_fn,
                        self.scale,
                        self.scale2,
                        base_seed + i,
                        self.worker_init_fn,
                        i
                    )
                )
                w.daemon = True
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            if self.pin_memory:
                self.data_queue = queue.Queue()
                pin_memory_thread = threading.Thread(
                    target=_utils.pin_memory._pin_memory_loop,
                    args=(
                        self.worker_result_queue,
                        self.data_queue,
                        torch.cuda.current_device(),
                        self.done_event
                    )
                )
                pin_memory_thread.daemon = True
                pin_memory_thread.start()
                self.pin_memory_thread = pin_memory_thread
            else:
                self.data_queue = self.worker_result_queue

            _utils.signal_handling._set_worker_pids(
                 id(self), tuple(w.pid for w in self.workers)
            )
            _utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

class MSDataLoader(DataLoader):
    def __init__(self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):

        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale
        self.scale2 = args.scale2

    def __iter__(self):
        return _MSDataLoaderIter(self)