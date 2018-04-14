1.实验环境：Ubuntu 14.04.5 LTS，512MB RAM, 2GB Disk

2.编译环境：gcc version 4.8.5 (Ubuntu 4.8.5-2ubuntu1~14.04.1)

3.编程语言：C++(100%)

4.采用自己编码实现的HMM训练，以及viterbi算法，5-fold交叉验证准确率如下：

(1) Precision = 0.98244

(2) Precision = 0.978036

(3) Precision = 0.982923

(4) Precision = 0.980025

(5) Precision = 0.981731

由于系统性能等因素，测试用时较长，一次验证要40min左右。由结果可以看出，准确率一般可以达到98%以上，准确率尚可。


注意：
1.SourceCode文件夹中为实验源代码，其中main.cc为运行主文件，实验所编写方法都在model.h文件中。测试时，请把data文件夹中trainingData.txt替换为原始语料。

2.限于系统性能等因素，5-fold交叉验证没有一次进行，而是分别进行，具体可参考代码中注释。
