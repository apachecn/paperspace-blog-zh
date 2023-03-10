# 快速行动，深入思考:研究是如何进行的@ Paperspace

> 原文：<https://blog.paperspace.com/how-we-do-research-at-paperspace/>

[2021 年 12 月 2 日更新:本文包含关于梯度实验的信息。实验现已被弃用。有关当前梯度资源的更多信息，请参见[梯度文档](https://docs.paperspace.com/gradient/explore-train-deploy/workflows)

先进技术组是 Paperspace 的一个以研发为中心的团队，由 ML 的工程师和研究人员组成。作为一个团队，我们有兴趣探索深度学习、数据工程、计算机系统和 UI/UX 方面的高级主题，并致力于构建智能应用程序。如果你对我们的工作感兴趣，考虑申请我们的[研究奖学金](https://jobs.lever.co/paperspace/7b9db8b1-36da-435c-9357-f737cb73e0ed)！

在本帖中，我们将以高级研究工作流程的形式，对高级技术小组(ATG)用于探索各种研究的工具和实践进行概述。我们的许多研究课题都位于深度学习和计算机系统等领域的交叉点。我们倾向于快速行动，处理雄心勃勃的计算密集型实验，由于我们通过 Paperspace 的梯度平台提供了许多非常有用的工具和强大的计算能力，我们可以追求涉及学术界更传统的研究小组有时会回避的主题的研究问题。

在这里，我们概述了研究工作流程的一般进展，我们发现这在我们处理的项目类型中非常有用。我们将讨论我们通常如何从最初的探索阶段前进，在这个阶段我们确定问题的范围并得到一些初步的结果。然后，我们将介绍如何在纸张空间云上扩展我们的实验。最后，我们将介绍我们如何将实验版本化，并跟踪研究议程的内部进展。

## 跟上 ML 消防水管

在机器学习领域，分享的想法和发表的论文数量之多几乎令人难以理解。要跟上每天冒出来的每一个新想法是非常困难的，其中许多当然是在基础突破上的渐进改进。在 ATG，我们的研究人员带着他们打算追求的特定想法加入团队，通常还有一些如何实现的想法。过去的想法和项目包括 GPU 内核编程、对抗性学习方案和神经架构搜索。

我们一直致力于引入与 ATG 的深度跨区域协作文化，并且经常会有想法转变，以纳入团队中其他感兴趣的成员或 Paperspace 中任何其他人的专业知识。我们也对许多 ML 理论家偏离的主题开放，包括可解释性、设计和人在回路系统。通过午餐和学习讲座、阅读小组会议以及允许任何人与其他人就一个有趣的项目展开对话的普遍开放的文化，可以分享新的想法。我们已经有软件工程师、项目经理和深度学习研究人员兴奋地讨论深度神经网络中模块化和修剪的含义。这是一次很棒的经历。聪明的人、天生的好奇心和高度协作的文化导致了许多不可思议的想法和项目在 Paperspace 形成。

## 探索一个想法:研究的面包和黄油

对于那些不熟悉研究的人来说，深入研究似乎是一项非常模糊和令人畏惧的任务，尤其是如果你的唯一经历是阅读论文和看到最终结果。实验的现实，尤其是在 ATG，是我们从一些实验结果的小延伸或问题开始的。我们可能会试图复制一篇论文的结果，或者在一个新的领域测试一个想法。自然地，当我们更好地理解作品的含义和基础时，有趣的想法和扩展就会出现。

当一个新的想法作为这个过程的结果开始形成时，我们把它缩小到经验上或理论上可测试的东西。尽可能缩小范围并保持简单是至关重要的，这样就可以清楚、直接地看到所产生的机制或期望的结果。作为一个例子，考虑我们可能想要测试一个新的修剪机制。我们将首先训练一个简单的完全连接的前馈架构，并在那里测试修剪机制，而不是在 ResNet 这样的复杂架构上测试新的修剪方案。然后，我们可以将 CNN 添加到探索性代码中，并在新架构上测试剪枝机制。

无论是重新实现另一篇论文的结果还是尝试一个你自己的新想法，目标都是在过程中有一个高水平的粒度和控制。在我们的团队中，我们发现梯度笔记本在这个过程中是一个非常有价值的工具。渐变笔记本允许我们使用预装了库和软件的容器，并向我们展示了一个 Jupyter 笔记本界面，该界面可以访问共享工作区，从而实现快速迭代和探索。

由于快速行动和测试许多小范围的可能性以获得概念性和经验性的理解是关键，我们非常频繁地使用这个特性。我们最近还在探索在笔记本电脑中使用 [Gradient SDK](https://docs.paperspace.com/gradient/gradient-python-sdk/gradient-python-sdk) ，使我们能够快速启动实验和更大的工作负载。如果我们生成了一个有用的结果，我们可以将它存储到共享的工作空间存储中，如果我们愿意，可以在更严肃的后续实验中使用它。此外，如果研究的某些内容是计算密集型的，即使我们将其范围缩小到概念验证实验，Gradient 也允许我们指定我们希望为我们的笔记本提供什么样的 GPU，这是我们在 Google Colab 等其他服务或本地 Jupyter 笔记本安装上无法做到的。

## 巨大的初始成果带来了大规模的后续实验。

哇哦。对我们新想法的初步探索产生了一些非常有趣的结果。我们的假设可能是正确的，那么现在呢？嗯，在很多领域，包括深度学习，你的方法或结果真的应该在一些更大的基准任务上进行测试。有些可能很小，但有些可能计算量很大。

较大的实验倾向于更加结构化，并且涉及相当程度的软件工程。它们需要更长的时间来设置，也要进行更严格的测试，以确保培训确实在进行。这通常是当我们团队的成员开始转向更有组织的代码库而不是单一的文件时。我们将开始使用设计原则，并开始真正记录一些工程决策。作为一名研究人员，当涉及到这些更大规模的实验时，笔记本界面开始变得有点缺乏，因为我不再花大部分时间通过小调整重新运行细胞并快速重新设计代码库。

在 ATG，我们可以访问 Gradient 的实验接口，这允许我们基本上将特定代码库的计算密集型运行视为作业。这些实验将运行我们指定的代码，并访问我们之前指定的共享工作区。结果是能够并行进行多个实验，并快速获得结果。我们还利用多节点特性和适当的分布式培训。Gradient 还自动解析关于我们的模型流程的统计数据，因此我们可以获得一些关于性能和其他重要指标的有用分析。

**关于工具**的快速说明。我们倾向于使用 Tensorflow，因为它有广阔的生态系统和对大型系统级实验的支持。我们也用过 Pytorch，发现它非常有用。

## 使用渐变 CI 体验版本控制

ML 研究中的一个正在进行的问题，也许是 CS 研究中的一个普遍问题，是决定如何将你的研究模型和实验版本化。作为研究人员，我们有时会发现调整代码库中的小值，如超参数值，可能会对我们的结果产生巨大的影响。但是，将学习率从 0.001 改变到 0.005 是否构成了我们正在跟踪的一个全新的实验？在 ATG，我们从我们的软件工程根源中获得灵感，并决定任何有用的提交变更都应该构成一个实验版本。毕竟，一个失败实验的成本肯定比跟踪许多增量实验的成本要高。Paperspace 的 [GradientCI 工具](https://docs.paperspace.com/gradient/projects/gradientci)可以跟踪变化，并在我们需要时自动运行这些变化作为实验。它还会以类似于 Gradient 客户端的方式，自动生成我们想要的各种指标的有用报告。

## 做研究没有正确的方法！

真的没有。研究过程应该是对你正在做的工作有意义的东西和让你的研究小组感到舒适和兴奋的东西的结合。在 ATG，我们结合了工程和研究背景，并发现我们上面提到的方法对测试 DL 和系统领域的大量有趣的想法非常有用。

从笔记本等灵活的工具转移到实验等更强大的界面，似乎遵循了我们正在进行的研究工作的自然流程，并允许我们利用软件工程最佳实践来提高工作效率。随着我们团队的成长，我们与全球其他世界级研究人员合作并建立更紧密的联系，我们希望进一步改善我们开放、合作和好奇的文化。

有兴趣加入 Paperspace 吗？你可以点击这里查看我们的招聘信息！