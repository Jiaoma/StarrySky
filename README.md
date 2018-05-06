# StarrySky
##一个用opencv和scipy实现的星空生成增强工具
程序主体是一个封装的类PerspectiveMatcher
##实现的功能有：
	1.图像拼接，将多张图片先通过surf算法采集星点坐标，然后对排在前stablenum个的星点进行匹配。匹配的方法是投影匹配
	星空的轨迹很诡异，在连续的拍照过程中，会呈现一个椭圆轨迹。简单平移，旋转不能满足要求。尝试过找圆心再旋转的办法，用到过随机梯度下降法，到头来发现效果最好的还是投影匹配。
	2.实现了类似photoshop的那种对于各个通道的曲线调整。
	3.锐化
	4.设置画面偏蓝
由于多个函数使用了默认参数，具体参数设置，看代码里面吧，名字应该能猜出是什么了。

示例:
```
    rootPath='/home/ljc/github/StarrySky/data'
    firstPath='/target/HQ_Gemini_Cancer'
    allPath='/major'
    testMatcher=PerspectiveMatcher(rootPath+firstPath)
    testMatcher.getAllImage(rootPath+firstPath+allPath)
    testMatcher.getStableStarXY()
    testMatcher.PerspectiveCombine()
    testMatcher.curveAdjust()
    testMatcher.sharpen()
    img,name=testMatcher.finalResult()
    cv2.imwrite('sr_{}.png'.format(name),img)
```
