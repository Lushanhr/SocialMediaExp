# SocialMediaExp

## 实验目的

评估一种社交媒体帖子文案优化方法（earlystop）是否能够提升帖子的人类真实互动倾向；具体而言，比较原始版本（Origin版本）与优化版本（earlystop版本）在人类浏览行为上的差异。

## 名词解释：

1.  feed：一个信息流
2.  post：信息流中的一条帖子
3.  target\_post：目标帖子，即我们要评测的帖子
4.  背景帖子：不是目标帖子的帖子，作为填充信息流的背景，构成环境

## 实验规模

*   被试人数：160人
*   每位被试看的feed量： 1 feed/人
*   每个feed 中 包含多少posts：20 posts / feed
*   每个feed 中有多少target\_post，其余是背景帖子：4 target\_posts / feed
*   总共的target\_post = 40

## 目标帖子和背景帖子的来源

- 目标帖子只能在测试集中选。

- 背景帖子可以从全集中选。

- 选取原则：
    1. 要求文案长度不能差别过大，无论是背景帖子还是目标帖子，具体而言，要满足下列约束：
        - 目标帖子 优化前后文案长度差别不大，假设长度为 L
        - 背景帖子的长度也在 L 附近
    2. 要求所有帖子的原始流行度相近，波动不超过30%

- 测试划分来源： /data/Lushanhr/popularity/CopyGRPO/data/ICIP/split_811_seed2026.json

- 原始图片文件夹：
/data/Lushanhr/popularity/data/ICIP/train_imgs

- 优化后的文案，自己找CopyGRPO 最新的earlystop 方法生成的文案

- 原始文案以及原始流行度就去原始数据集里看就行：/data/Lushanhr/popularity/CopyGRPO/data/ICIP/merged.pkl



## 怎么分配


### 单个 Feed 结构

- 总帖数：20

- 目标帖：4 个

- 背景帖：16 个（全部 feed 共用）

### 目标帖与组合

- 共 40 个目标帖，分成 10 组，每组 4 个

- 每 4 个 目标帖 构成 2 的 4 次方 = 16 种组合，意思是 一个目标帖的 原始文本 和 优化后的文本 不能出现在同一个 Feed 中

- 每组 16 种组合，共 10 组，每组 16 种组合，共 160 种组合，160个feed

- 比如 ABCD 四个目标帖，这4个目标帖形成的16 个 feed是： A-原 B-原 C-原 D-原 + 背景帖子、 A-原 B-原 C-原 D-优化  + 背景帖子 ··· A优化 B-优化 C-优化 D-优化 + 背景帖子


### 如何填写social_feed_test.csv

- 填写这个是最终目的

- 字段说明：
列名	必填/选填	填什么	举例	特别注意
condition	必填	这是"组别身份证"。同一组的 20 行必须填一模一样的值。	combo_1_0001、combo_1_0002	这是最关键的一列。我们一共会有 160 个不同的组别代号。
doc_id	必填	帖子的编号。不同组的同一张图片，编号要一致。	1、2、3	目标帖子填 1-4；背景帖子填 5-20。
text	必填	帖子的文案内容。	原始文案 / 优化文案	注意：根据 condition 的设计，判断这里该填"原始版"还是"优化版"。
media	必填	图片的网络地址（URL）。	示例：https://media.githubusercontent.com/media/Lushanhr/SocialMediaExp/refs/heads/main/test_images_811/36830698624.jpg
likes	必填	该帖子的点赞数。	低范围随机值
username	必填	发帖人的昵称。	随机，示例：TravelLover	
handle	必填	发帖人的账号（@后面的部分）。	随机，示例：'@TravelLover'
user_image	必填	发帖人头像的网络地址。	可以使用默认头像，也可以每行不同，参考现在的模版。
datetime	必填	发帖时间。	01.06.24 12:00	格式必须严格一致：日.月.年 时:分
reposts	必填	转发数。	5	低范围随机值 。
replies	必填	评论数。	2	低范围随机值 。
sequence	选填	控制帖子在 Feed 里的位置。	留空	不要填任何数字，留空即可。系统会自动随机排序。
其余列	选填	如 alt_text 等	留空	保持空白即可，不影响运行。

