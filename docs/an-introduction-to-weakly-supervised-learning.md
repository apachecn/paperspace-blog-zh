# 弱监督学习导论

> 原文：<https://blog.paperspace.com/an-introduction-to-weakly-supervised-learning/>

**简介**

在过去的几年里，机器学习经历了许多变化。对无监督和半监督学习的关注一直在增加，但一种称为弱监督学习的新技术最近引起了研究人员和行业专业人士的关注。

弱监督学习中训练模型不需要任何标记数据。然而，这是有代价的:与其他类型的监督学习方法相比，准确性受到影响。在本文中，将详细研究这项新兴技术。

**监督和非监督学习**

简而言之，机器学习可以采取两种方法之一:监督学习和非监督学习。前者用于使用已知的输入和输出数据来训练模型，以预测未来的输出，而后者用于揭示输入数据的内在结构中的隐藏模式。

**监督学习**

监督学习是指使用标记数据来训练机器(或深度)学习算法，目标是对未来发生的事情进行预测。训练好的算法的成功依赖于用高质量标签完全注释的大数据集。在监督期间使用这种数据收集的过程被称为强监督学习。

在监督学习中，使用标记数据训练模型，标记数据包括原始输入数据和模型输出。数据分为训练集和测试集，训练集用于训练我们的网络，测试集用于生成预测结果或确定模型正确性的新数据。

有两种类型的监督学习:

分类——分类问题利用算法将测试数据分类到特定的类别中。一些流行的分类算法包括决策树、支持向量机、线性分类器等。

回归-回归是另一种类型的监督学习，它使用算法来理解自变量和因变量之间的关系。回归模型对于通过使用不同的数据点来预测数值非常有用。有几种常见的回归方法，包括逻辑回归、线性回归和多项式回归。

**无监督学习**

当比较监督学习和非监督学习时，最重要的区别是前者利用了标记的输入和输出数据，而后者没有。无监督学习是一个通用术语，指的是各种结果不确定的机器学习，并且没有教师来教授学习算法。无监督学习只是向学习算法显示输入数据，并要求它从中提取信息。

无监督学习利用机器学习算法从无标签数据集中推断模式。这些算法擅长发现数据中的隐藏模式，但不需要人工干预。其检测数据相似性和差异的能力使其成为图像识别数据分析和图像识别等的良好选择。

无监督学习有以下几种类型:

聚类-聚类是一种将项目排列成簇的技术，这种方式使得具有最大相似度的对象留在一个组中，而与另一个组中的对象具有很小或没有相似性。

关联-关联规则是一种无监督学习技术，用于识别大型数据库中变量之间的联系。这是另一种无监督学习方法，采用不同的规则来确定给定数据集中变量之间的关系。

**什么是弱监督学习？**

术语“弱监督学习”指的是各种各样的学习过程，其目的是在没有太多监督的情况下构建预测模型。它包括灌输领域知识的方法，以及基于新创建的训练数据标记数据的函数。

当所提供的数据准确地覆盖了模型预期的领域并且按照模型的特征进行组织时，机器学习模型如预期的那样执行。因为大多数可访问的数据都是非结构化或低结构化的格式，所以您应该利用弱监管来继续对这种数据进行注释。换句话说，当数据被注释但质量差时，弱监管是有帮助的。

弱监督学习是一种基于新生成的数据建立模型的技术。它是机器学习的一个分支，使用嘈杂、受限或不准确的来源来标记大量的训练数据。弱监督涵盖了广泛的方法，在这些方法中，模型使用部分的、不精确的或不准确的信息进行训练，这些信息比手动标记的数据更容易提供。

为什么我们需要弱监督学习？

监督学习方法从大量训练实例中建立预测模型。每一个都使用地面实况输出进行标记。虽然现有的方法非常有效，但重要的是要注意，由于与数据标记过程相关的高成本，机器学习算法最好在弱监督下运行。

此外，虽然监督学习通常比弱监督学习使用更多的标记数据来提高性能，但当使用较少的标记数据进行学习时，我们可能会看到性能下降。因此，研究弱监督学习是非常必要的，即使在弱监督数据的情况下，弱监督学习也有助于提高性能。

对于解决训练数据短缺的问题，弱监督比其他方法更有效且可扩展。弱监督使得有许多输入帮助训练数据成为可能。获得手工标注的数据集可能是昂贵的或不切实际的。为了建立一个强大的预测模型，尽管不准确，还是使用弱标签。

弱监督可以让你对训练数据进行编程，减少手动标注的时间。它最适合需要处理未标记数据的任务，或者您的用例允许弱标签源的情况。

**弱监督学习技术**

弱监督学习方法通过仅使用部分标记的模型来帮助减少训练模型中的人工参与。它介于完全监督学习和半监督学习之间。这是一种使用带有噪声标签的数据的方法。这些标签通常由计算机通过使用试探法将未标记的数据与信号组合来创建它们的标签而生成。

在弱监督学习中，算法从大量弱监督数据中学习。这可能包括:

不完全监督(例如，半监督学习)。

不精确的监督，例如，多实例学习。

不正确的监督(例如，标签噪音学习)。

**监督不彻底**

当只有训练数据的子集被标记并用于训练时，就会发生不完全监督。这种类型只标记一小部分训练数据。这个数据子集通常被正确和精确地标记，但这不足以训练一个有监督的模型。

不完全监督有三种方法:

主动学习-主动学习是一种半监督学习，其中 ML 算法从更大的未标记数据集中获得一小部分人类标记的数据。该算法对数据进行分析，并做出具有一定置信度的预测。

半监督学习——半监督学习是一种结合使用有标签和无标签样本的学习方法。然后，模型必须从这些示例中学习以做出预测。

迁移学习-在机器学习中，迁移学习涉及存储和应用从解决一个问题中获得的知识到另一个问题中。例如，从学习如何识别汽车中获得的知识可以应用于公共汽车的识别。

**监督不准确**

不精确监督使用标签和特征(即元数据)来识别训练数据。当训练数据包括不如期望的精确的标签并且仅与粗粒度标签一起使用时，发生不精确监督。

**监督不准确**

当可用的标签不一定是地面真相时，不准确的监督被应用。顾名思义，不准确的监督包含错误，某些地面真相标签是不正确的或低质量的。

这通常发生在众包数据时，当有干扰、错误或难以对数据进行分类时。目的是收集可能贴错标签的实例并纠正它们。当训练数据中存在一些带有错误的标签时，就会出现监督不准确的情况。

**弱监管应用**

监管不力与特定的监管任务或问题无关。相反，如果训练数据集的注释不充分或不完整，您应该利用弱监督来获得具有优异性能的预测模型。您可以利用文本分类、垃圾邮件检测、图像分类、对象识别、医疗诊断、金融问题等方面的薄弱监管。

**用于目标定位的弱监督学习**

识别图片中一个或多个项目的位置并围绕其范围绘制边界框的过程被称为对象定位。目标检测结合了图像定位和图像中一个或几个目标的分类。

弱监督对象定位是一种在不包含任何定位信息的数据集中查找项目的策略。近年来，由于其在下一代计算机视觉系统开发中的重要性，它受到了相当大的关注。分类模型的特征图可以通过仅用图像级注释简单地训练它而被用作用于定位的得分图。

**监管框架薄弱**

有几个薄弱的监管框架，例如:

是一个弱监管框架，可作为一个开源项目。您可以利用少量已标记数据和大量未标记数据在 Python 中构造标注函数。

对于深度神经网络来说，Astra - Astra 是一个糟糕的监督框架，它为大规模、昂贵的标记训练数据不可行的任务创建弱标记数据。

con wea——con wea 是一个为文本分类提供基于上下文的弱监督的框架。

snu ba-snu ba 生成启发式算法来标记数据子集，然后迭代地重复该过程，直到覆盖了大量未标记的数据。

flying squid——flying squid，一个基于 Python 的交互式框架，允许你从多个嘈杂的标签源自动创建模型。

**总结**

无监督学习和有监督学习是两种最流行的机器学习方法。弱监督介于两个极端之间:半监督学习和完全监督学习。

当训练数据的注释不完全或不足以创建预测模型时，可以使用弱监督。它可用于图像分类、物体识别和文本分类。