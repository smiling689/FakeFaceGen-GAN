# MiniProject: FakeFaceGen

#### Generator's structure

![gen](https://www.paddlepaddle.org.cn/documentation/docs/zh/_images/models.png)

#### Discriminator's structure

![dis](https://i-blog.csdnimg.cn/blog_migrate/60ff6330d1f200ee3bdfa76f5e6a4ea0.png#pic_center)

## Dataset

我们使用 [Celeb-A Faces](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 数据集。

## File Structure

### `preprocess.py`

用于预处理数据集，把图片缩放到 64x64 的大小。预期处理后的目录为 "img/processed/"

### `dataloader.py`

用于加载数据集，在本次项目中，你不需要显示调用这个文件。值得注意的是，加载的数据集不是原始数据集，而是经过 `preprocess.py` 处理后的数据。

### `network.py`

用于构建 DCGAN 的网络结构。也是你需要完成的部分。

### `train.py`

用于训练网络，其中训练完成的参数会保存在 "generator.params" 中。

### `generator.py`

从 "generator.params" 中加载训练好的参数，并生成图片。

### workflow

实际上你需要先运行 `preprocess.py`，然后运行 `train.py`，最后运行 `generator.py`。

`preprocess.py` 只需要运行一次（在不修改 `preprocess.py` 的情况下）。