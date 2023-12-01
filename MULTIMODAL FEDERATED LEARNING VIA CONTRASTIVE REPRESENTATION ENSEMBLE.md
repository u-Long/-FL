### MULTIMODAL FEDERATED LEARNING VIA CONTRASTIVE REPRESENTATION ENSEMBLE

==将来自不同模态的信号统一到同一向量空间中==

#### Abstract

设计了一种全局-局部跨模态集成策略来聚合客户端表示。

为了减轻由多模态差异（模态差距和任务差距）引起的两个前所未有的异质因素引起的局部模型漂移，我们进一步提出模态间和模态内对比来规范local训练，这补充了缺失信息的单模态clients并规范local client达成global consensus。

解释：

模式差距：在单一模态（例如图像）下接受训练的单模态客户在训练过程中从未见过来自其他模态（例如文本）的数据，因此缺乏识别另一种模态的能力。
任务差距：不同的客户端可以针对不同的任务进行训练，例如，用于图像分类任务的单模态客户端和用于图像文本检索任务的多模态客户端。
这两个差距会导致model drift。

#### relate work

采用FedAvg框架对每种模态用同质模型（homogeneous models）：Xiong et al., 2022; Zhao et al., 2022; Liu et al., 2020

clients间的model drift问题：Karimireddy et al., 2020

FedET等KD框架（服务端大模型，客户端相对小模型）是基于logits传递和整合knowledge的，很难扩展到多模态环境。

应用KD到FL：Lin et al., 2020; Itahara et al., 2021; Wu et al., 2021

允许异构客户从aggregated consensus中dk，但不训练服务端：Li & Wang (2019) 

允许更大的服务器模型，但服务器在客户端之间不进行选择，从而导致达成不良共识影响性能：FedGKT

通过ensemble knowledge transfer在FL中训练大的server端模型（聚合策略是logit，且学习的大模型仅限于分类任务）：Cho et al. (2022) and Cheng et al. (2021)

![QQ截图20231124143724](../../../typora/typora图像集/QQ截图20231124143724.png)

本文设计了一种基于 KD 的多模态联邦学习框架 CreamFL（多模态 FL 的对比表示集成和聚合），它同时利用跨异构客户端的单模态和多模态数据，通过==representation-level ensemble knowledge transfer==。全局模型通过在不同客户端网络的公共数据集上交流私有知识来向客户端学习，而不泄露私有模型和数据。 CreamFL 在服务器和客户端之间传输公共数据的低维表示==（low-dimensional representations）==，这些数据通常是上下文相关的，并且适用于比 logits 更复杂的任务。

聚合策略：

为了有效地聚合从异构客户端传输的表示，我们提出了一种全局-局部跨模式对比聚合策略，1）通过将局部表示与全局表示进行对比来过滤掉漂移的异常值； 2）通过与其他模式的表示进行对比，挑选出与配对伙伴更匹配的优秀候选人。

解决modal drift：

两个对比目标（inter-modal，intra-modal）规范local train。inter-modal就是用公共数据及进行local train。intra-modal就是将每种模态中的local表征与相应的global表征进行对比

![QQ截图20231122103955](C:\Users\SYL\AppData\typora\typora图像集\QQ截图20231122103955.png)

$M$多模态客户端，$I$图片客户端，$T$文本客户端。用集成知识蒸馏训练global model $f_s(·;w):R^n ->R^d$，$w$是模型参数，$n$是输入数据的维度，$d$是输入数据的提取特征。

$I_p={(x_p^k,y_p^k)}_{k=1}^{|I_p|}$表示图片客户端的私有数据集（第p个客户端的第k个训练样本），多模态和文本同理。

server端和client端都可以访问公共数据集$P={(i^k,t^k)}_{k=1}^{|P|}$，假设每个客户端基于本地数据采用一个小模型$f_c(·;w_c):R^n ->R^d$。

在训练期间，客户端首先接收公共数据的全局表示，并通过模态间和模内正则化执行表示学习的多个局部步骤（第 3.2 节）。然后，客户端根据自己的模态生成公共数据的表示并将其传输到服务器。为了消除不同模式和客户端之间的偏差，服务器有选择地聚合表示（第 3.3 节）。最后，服务器从聚合的表示中执行知识蒸馏，并将其经过训练的表示传输回客户端。完整的算法在算法 1 中提供。

#### LOCAL TRAINING VIA CONTRASTIVE REGULARIZATION (LCR)

1. inter-modal对比，以补充缺失模态的信息。
	以图像客户端c为例，每轮通信开始时，接收公共数据$t_{global}^j\in R^d, j=1,...,|P|$，对本地$i^k$generate表示$i_{local}^k\in R^d$，注意此时的i可能有很大的偏差。==假设全局多模式服务器已经部分学习了图像和文本表示之间的共享表示空间。==

	对比损失：![QQ截图20231124160235](../../../typora/typora图像集/QQ截图20231124160235.png)

2. intra-modal对比，以解决模型漂移。
	以图像客户端c为例，每轮通信开始时，接收全局图像表示$i_{global}^j\in R^d, j=1,...,|P|$，将本地的$i_{local}^k\in R^d$$i_{local}^k$与对应的全局$i_{global}^k$进行对比。同时遵循MOON，这里再加一个和上一轮的本地表示$i_{prev}^k$的负对比.

	对比损失：

	![QQ截图20231124161614](../../../typora/typora图像集/QQ截图20231124161614.png)

最终对比损失：

![QQ截图20231124161721](../../../typora/typora图像集/QQ截图20231124161721.png)

这里前一项是原始local训练的目标损失，后面是我们的指标。

#### GLOBAL-LOCAL CONTRASTIVE AGGREGATION (GCA)

以聚合第c个客户端的第k的图像表示$i_{local}^{(k,c)}$为例，让它更好地匹配其对应的全局表示$t_{global}^k$，并远离其他表示$t_{global}^j, j!=k$。

得分：

![QQ截图20231124163946](../../../typora/typora图像集/QQ截图20231124163946.png)

我们使用 softmax 进行归一化，并将局部图像表示 $i_{local}^{(k,c)}$ 聚合为：

![QQ截图20231124164102](../../../typora/typora图像集/QQ截图20231124164102.png)

知识迁移：

聚合后，服务端模型通过最小化下面的$l_2$距离进行知识蒸馏。

![QQ截图20231124164348](../../../typora/typora图像集/QQ截图20231124164348.png)

![QQ截图20231124164821](../../../typora/typora图像集/QQ截图20231124164821.png)

#### APPENDIX

选择具有 50,000 个图像文本对的 COCO 随机子集（Lin 等人，2014）作为公共多模态数据。 CIFAR100、AG NEWS 和 Flicker30k 分别用作图像、文本和多模式客户端的私有数据集。 

CIFAR-100（Krizhevsky 等人，2009）由 100 个类别的 50,000 个彩色训练图像组成，每个类别有 500 个图像。 AG NEWS (Zhang et al, 2015) 包含来自 4 个类别的 120,000 个训练句子。 
Flicker30k（Plummer 等人，2015）包含从 Flicker 收集的 31,000 张图像，以及每张图像 5 个标题，即总共 155,000 个图像文本对。
对于跨模态检索，我们遵循 Karpathy & Fei-Fei (2015) 并报告 MS-COCO 5K/1K 测试集上的 Recall@K 结果，该结果测量在 top-K 结果中找到正确项目的次数百分比。对于视觉问答，我们使用 VQA v2.0 数据集（Goyal 等人，2017 年）并报告 3,000 个最常见答案的准确性。