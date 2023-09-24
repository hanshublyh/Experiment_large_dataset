1.分别运行run_a1.sh, run_a3.sh
2.main.ipynb是生成数据用的
3.deivce是所有文件共用的参数 指定gpu
4.5_40不运行 是我自己测试并且生成数据用的 
5.每种文件之后的para1.txt para3.txt是单和多智能体的参数 从80_200-->150_500基本一致 step_r不一样
6.在para1.txt para3.txt里 batch_size统一设置为256
7.para1和para3本来应该在部分参数不一样的 为了对比全设置为一样的了
8. 不用resnet 大规模的数据集 input是224 * 224 / 448 * 448