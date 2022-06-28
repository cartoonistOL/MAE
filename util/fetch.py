r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""


class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
            # 多返回一个time list
            # 这里路径表示不一致，windows可以用"\\"，在linux上要用"/"
            time = [self.dataset.imgs[idx][0].split("/")[-1].split("_")[0] for idx in possibly_batched_index]
            # 返回location list
            # 利用曼哈顿距离作为位置信息
            centerls = [self.dataset.imgs[idx][0].split("/")[-1].split("_")[2] for idx in possibly_batched_index]
            rowcolumnls = [self.dataset.imgs[idx][0].split("/")[-1].split("_")[-1].split(".")[0] for idx in possibly_batched_index]
            locationls = [(int(centerls[i][0:2]) - int(rowcolumnls[i][0:2]))**2 + (int(centerls[i][2:4]) - int(rowcolumnls[i][2:4]))**2 for i in range(len(possibly_batched_index))]

        else:
            data = self.dataset[possibly_batched_index]
            time = self.dataset.imgs[possibly_batched_index][0].split("/")[-1].split("_")[0]
            centerls = self.dataset.imgs[possibly_batched_index][0].split("/")[-1].split("_")[-2]
            rowcolumnls = self.dataset.imgs[possibly_batched_index][0].split("/")[-1].split("_")[-1]
            locationls = (int(centerls[possibly_batched_index][0:2]) - int(rowcolumnls[possibly_batched_index][0:2])) ** 2 + (
                        int(centerls[possibly_batched_index][2:4]) - int(rowcolumnls[possibly_batched_index][2:4])) ** 2
        return self.collate_fn(data),time,locationls
