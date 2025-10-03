# Data Layout & Danbooru2019 Instructions

This project expects a raw portrait dataset under `data/raw/danbooru2019/`.

## Option A — Danbooru2023 subset (Hugging Face)
Danbooru2023 is huge (~8 TB). For experimentation you can fetch a single shard,
for example `original/data-0000.tar`.
https://huggingface.co/datasets/nyanko7/danbooru2023/resolve/main/original/data-0000.tar
```shell
tar xf data-0000.tar -C data/raw/danbooru2019/
```
