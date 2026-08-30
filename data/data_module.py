# DataModule for initialization datasets and dataloaders (and distributed samplers)

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from typing import Callable, Optional


class DataModule:
    r"""
    DataModule manages datasets, data_loaders, initializes preprocessor,
    and distributed samplers.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        trainsets: Optional[list[Dataset]] = None,
        evalsets: Optional[list[Dataset]] = None,
        train_collate_fn: Optional[Callable] = None,
        eval_collate_fn: Optional[Callable] = None,
        is_distributed: bool = False,
    ):
        r"""
        Args:
            trainsets: List of datasets, they will be converted into ConcatDataset.
            evalsets: Used as evaluation datasets.
            is_distributed: Configure whether use distributed sampler in dataloader.

        NOTE: Raise error if both trainsets and evalsets are None.
        NOTE: DataModule do not provide dataset spliting utils, recommand split
            trainsets and evalsets in advance.
        """

        if trainsets is None and evalsets is None:
            raise ValueError(f"One of trainsets or evalsets should be given.")

        # These must be access via @property methods
        self._trainset = ConcatDataset(datasets=trainsets) if trainsets is not None else None
        self._evalset = ConcatDataset(datasets=evalsets) if evalsets is not None else None
        self._train_loader = None
        self._eval_loader = None

        # Public attributes
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_distributed = is_distributed

    @property
    def trainset(self) -> Dataset:
        if self._trainset is None:
            raise ValueError(f"trainset is None, please configure it at initialization stage.")
        return self._trainset

    @property
    def num_train_samples(self) -> int:
        return len(self._trainset) if self._trainset is not None else 0

    @property
    def evalset(self) -> Dataset:
        if self._evalset is None:
            raise ValueError(f"evalset is None, please configure it at initialization stage.")
        return self._evalset

    @property
    def num_eval_samples(self) -> int:
        return len(self._evalset) if self._evalset is not None else 0

    @property
    def train_loader(self) -> DataLoader:
        if self._trainset is None:
            raise ValueError(f"trainset is None, please configure it at initialization stage.")

        if self._train_loader is None:
            sampler = None
            if self.is_distributed:
                sampler = DistributedSampler(self.trainset)
            self._train_loader = DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
                shuffle=sampler is None,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.train_collate_fn,
            )
        return self._train_loader

    @property
    def eval_loader(self) -> DataLoader:
        if self._evalset is None:
            raise ValueError(f"evalset is None, please configure it at initialization stage.")

        if self._eval_loader is None:
            sampler = None
            if self.is_distributed:
                sampler = DistributedSampler(self.evalset)
            self._eval_loader = DataLoader(
                self.evalset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
                shuffle=sampler is None,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.eval_collate_fn,
            )
