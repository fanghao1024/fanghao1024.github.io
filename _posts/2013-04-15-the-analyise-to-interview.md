---
layout: post
title: "这一周的实习面试总结"
description: "这一周参加了三个公司的面试，有不少感触，把经历和感想写一下"
author:     "Hao"
category: false
tagline: " 浮华过去了，回归平静"
tags: [interview , thinking]
---

<div class="blogcontent">

	<p>这个博客是3个月之前弄的，一直想往上面加东西，但是一直很忙，没有时间弄。当初挺频繁地写日志各种吐槽，完全是因为太空了，现在才明白，经常写文章的人，要么是因为太空了，要么是因为太有毅力了。</p>
	<p>各大公司的实习生招聘也大多密集地在四月份进行，这一周我就面了三个公司，到今天下午终于全面完了，闲话少说，做个总结吧。</p>
	
	<p><B>Yahoo北京研发中心</B></p>
	    <p>这个算是雅虎美国在中国独立机构，和阿里巴巴的中国雅虎没有什么关系，坐落在清华的东边，当时应该是在大街网上随便点击申请的，然后在清明节放完假以后，Yahoo的HR就约了上周二上午10点的面试。</p>
<p>进入前台就感觉有点不舒服，因为Yahoo的标志，是紫色字体白色背景，感觉怪怪的，很不喜欢这种色调（貌似隔壁的代表色也是紫白。。。）。接下来进行了两轮的面试。首先进来一个比较年轻的工程师，也没让我做自我介绍，也没有问我简历上的东西，就让我做了3道算法题。第一道题是给一个字符集合，然后把这个集合的全排列打印出来。好吧，这是道很简单的问题，不过其实我挺讨厌组合排列方面的题目，因为觉得麻烦，平常都是直接调用STL函数这样的库函数来实现的。但麻烦归麻烦，还是得自己实现，用了递归的方法，最后检查代码花了一些时间，我漏了个返回，其实在检查的时候我就已经发现我代码的一个致命缺陷了，不过他当时没有发现；第二题我在面Amazon的时候也碰到了，题目给出了一个背景，买卖股票，给你连续N天的股票的价格，然后你得确定一个买入的日期和卖出的日期，使得其收益是最大的，simple problem, O(n)的算法扫一遍，各种记录；第三道题目是给一刻二叉搜索树，然后给一个值，输出这个树里面的结点和这个值的差的绝对值的最小值，一道类似二分查找的题目，确定左子树和右子树各自的范围，然后判断给定值的位置，然后放到子问题中解决即可。然后差不多将近1个小时，就来了第二个面试官。他先让我做了自我介绍，然后问了我简历里面感兴趣的一些东西，我发现我简历中关于GPU并行计算的东西经常引起面试官的兴趣，不管是当初的百度面试，还是后来的网易游戏、Amazon面试，都被提到了，而其中的关于中药黄芩的那个有关模式识别和机器学习的项目，除了Amazon的俩面试官问了，其他人压根就没提及过，然后我介绍了我正在自己做的一个操作系统，很显然的，他狠狠问了一把操作系统的东西。面试压根就是个不公平的流程，面试官就问他所知道的那部分，谁知道他无意中看过啥偏门的东西，要把人问倒很容易的么，我觉得有时候也得踢踢馆，什么时候玩玩反客为主考倒面试官的游戏。然后他出了一道代码题，给一个链表，如１->2->3->4->5->6->……，作为输入，输出为两个指针：1->3->5->….，2n->2(n-1)->…->6->4->2，数字代表的是结点在原链表中的位置。超简单的题目有木有，于是我就随手写了，果然就大意了，出了漏。差不多12点了，然后面试就结束了。</p>
        <p>结果：这边要实习生马上来工作，我希望暑期开始，然后进去做的是类似广告推荐的东西，我不是很感兴趣，另外还有个不足为外人道的原因，所以我就不去了，Yahoo问了我能实习的时间，之后就给我发了封拒信，thank you for applying 神马的。。。我以后肯定去Yahoo工作，如果它还活着的话。。。</p>
        <p>吐槽：算法出难点会死啊，这些可以秒杀的题目很考我的演技啊亲！</p>

<p><B>网易游戏</B></p>
<p>这个是在网易实习生招聘官网上投的简历，先是进行一轮电话面试，一个带着浓重广东口音的HR慢条斯理地问问题，有做个自我介绍啊，你讲讲你认为你自己做过的最好的项目啊，你在做项目的时候有没有遇到什么很困难的问题啊，你是用什么方法来解决的啊，你觉得你的优缺点是什么啊，我看你之前在百度实习过，那么你觉得百度这个公司怎么样啊。除了最后一个问题，其他问题我都事先准备了答案，就照着念了，最后一个问题当时我就好纠结，该怎么回答呢，如果我说百度很好很好，那么她会不会问，既然百度那么好，你为啥申我们公司呢，如果说百度缺点不少，那她会不会想我既然这么对待之前实习的公司，以后会不会也这样对他们公司啊。后来想想，这个和小时候纠结到底去北大还是去清华一样没有意思，人家其实就是唠唠，貌似没有听说过电面就挂的。</p>
<p>清明节结束网易就通知了在文津国际宾馆的实习生见面会，签到的时候一溜儿的都注明了北大清华的硕士博士，开场前放的是他们正在开发的游戏的demon，其中性感女郎伴着音乐跳舞的游戏demon很吸引我的注意，其它的几个demon我觉得做的也一般。然后是一个副总，一个技术总监，还有战略方面的，虚拟世界架构方面的负责人讲了讲他们公司的一些理念啊啥的。虽然网易游戏给的薪水很高，但除此之外好像没有能吸引我的东西。比如网易偏向于低调的做事风格，而我更喜欢比较强势高调一点的公司；比如网易游戏并不倾向于扩大而是实施所谓的精英化培养，追求小而精，他以为他是基金啊，而我更喜欢帝国式的公司，因为对外更强势么。</p>
<p>然后第二天黄昏的时候进行面试，首先呢给你两道题，然后让你在半个小时之内选择一道答之。它要求保密题目，那我就不详说什么题了，算法不要求什么技巧，但是看着就让人烦。半小时后，就去了12层的一个房间，2对1的面试，有一个是见面会的那个技术总监，是个隔壁的博士，先问了下笔试题目的思路，然后照着简历上写的一项一项往下问，看什么时候能把你问倒，好吧，在下本科学校计算机实力确实很弱，所以很多基础的东西确实不像工科学校毕业的那么扎实，但是你可以问算法题么，你问几道其它的算法题会死啊，之后竟然一道都没问。本科本来就是通识教育，应该多接触专业的各个方向，所以我接触了很多领域的项目，那当然不可能太深入的么，被问倒是很正常的，但一脸意味深长的笑算神马。。。</p>
<p>于是我突然萌生了一个想法，虽然我不会去网易（一个原因是它在广州和杭州，而我是不会离开北京了，一个原因是它做的不是我感兴趣的东西），但是来年等我在某个领域大成（比如算法方面）一定要来踢踢馆子。。。</p>

<p><B>Amazon</B></p>
<p>今天下午面的Amazon，真心远啊，北大在大概西北四环的地方，丫在东南四环啊，转半个北京城才能到啊，单程就花费将近100分钟，我可以保证没有堵车，因为我坐的是地铁。</p>
<p>Amazon的实习生面试也是两面，但是第一面的大部分时间面了项目，我不知道这个面试官是多寂寞了，竟然详细地问了我那个中药黄芩模式识别的破项目！其它项目的问题大同小异，关于原理啊，有啥难点啊之类的。然后问了面向对象里protect的详细内容，好久没用，忘了。。。然后又问了static是个啥特性。好在这个我前几天看到过了。然后出了个算法题，给一个整数数组，和一个值，找出这个数组里和等于这个数的整数对。我刚开始想了一个先排序，然后设俩指针根据指针所指数的和的大小，一个指针从前往后扫，一个指针从后往前扫的，O（nlogn）的复杂度，然后他问有没有更快的算法，Ouch！Bit-map呗，O（n），然后写代码。写完后，他很认真地批评了：你的程序没有防御性，如果怎么怎么了你程序就崩溃了，如果怎么怎么了就崩溃了，要我补上防御性代码，我当然没有完全补全，然后他语重心长地跟我说，孩子，在工业项目里，这些防御性代码就算不比算法重要，也和算法一样重要啊。我当时心里就闪过一个念头：哥不要当工程师了，哥要当科学家。。。</p>
<p>然后又来一个面试官进行第二面，他也先问了项目，那个中药黄芩的项目又被问了！然后他详细问了我现在在做的项目，并详细问了我马尔科夫逻辑网络的东西。然后问我网络编程的熟练度如何啊，会什么网络编程语言啊，java的熟练程度呢？会用perl吗？正则表达式呢？知道数据库第三范式是什么，能详细说说吗？给你个函数 fool(){int a=0;double b=1.1;ha(a,b);a=0;} ha(int a,double b){},当执行到fool()里的ha函数的时候，内存的栈里的情况是怎样的？除了这些还有呢？不应该在组成原理里面讲到的吗？你觉得亚马逊这个前端界面如何，有啥需要改进的地方？你会用gdb调试吗？你用的是啥编译器啊？难道你编Java用的也是VS？哦，Eclipse啊，那你用得比较熟咯？对了，把这个页面的这几行字改大试试。我们接下来做道算法题吧，就是在Yahoo北京研发中心的拿到股票题，我就加上些所谓防御性代码的条件判断，然后面试就结束了。</p>

<p>总结：之前在知乎上有问过问题，然后一个来自Mountain View, California 的人回复了我，我觉得很有道理，原文如下：谢邀。就我个人遇到过的应届生情况来看，大部分应聘者基本功都比较成问题，要想提高算法能力，先夯实基本功才是正道。基本功足够扎实之后，无论你是做面试提还是刷 ACM 题，或者看算法导论，都能起到效果。说实话绝大部分面试题对算法的要求不高，不要太过于高估算法对于面试的作用。</p>
<p>确实如此，这3次面试答得不太好的都不是算法方面的问题，而且问的算法题都很简单，而回答得不太好，或者比较狼狈的往往都是一些很基础的问题。正像沃伦•巴菲特的那句名言：别人贪婪的时候我恐惧，别人恐惧的时候我贪婪。在这个人心浮躁的环境里，跟着一起浮躁只会让人疯狂。现在要做的，就是真正地静下心来踏踏实实地积淀自己的能力，夯实基本功，贵精不贵多，真正地深入精通某个领域，当人们提起这个领域的时候就会想到你。</p>
<p>现在在想要不要转博，做个苦逼的博士，诚然国内博士质量水，但是关键还是看自己，如果每天就是在看看电影电视剧然后开会前赶赶报告什么的话，那理所当然很水，而发自内心由内心驱动来学习充实自己，那一定是另一种结果了。</p>
<p>当然离申请转博的时间还长，可以慢慢考虑。</p>
<p><B>最近的计划：</B></p>
<ul>
<li>		现在重新启动了看TAOCP，不要浮躁，慢慢看，多久能看完就用多久，欲速则不达，也认真做上面的习题，如果做不出，也至少要有个思考的过程；</li>
<li>		继续研究关于人工智能和机器学习方面的资料，我认为这是个很有趣而且也很有前景的方向，通过开发人工智能进一步解放机器的生产力，太激动人心了！</li>
<li>            继续开发之前自己做的那个操作系统，在做完了内存管理图形显示等相关内容后，接下来要做的就是多进程的调度以及文件系统了；</li>
<li>	        夯实基础，对于本科时因为不认真而不是很清楚的专业知识，如C++的细节啊计算机体系结构相关的东西，慢慢弥补吧</li>
<li>	        不要只局限于纯技术专业领域，加强艺术修养（继续研读《艺术通史》）以及人文修养（继续看《失控》以及相关系列资料）</li>
 <li>           尝试参与开源项目，工业界嘲讽学术界的一个弊病就是代码太小儿科，我不太想出去实习，而是想真正地积淀点长远有用的东西，但是太脱离实际的话有点类似闭门造车，开源项目也算理论联系实际的一个途径吧。</li>

<p>	   引用之前那句很经典又很烂大街的话：以我们的努力程度之低远远还没到拼智商的程度。压抑不住浮躁的心，往往因为耐不住寂寞。</p>
<p>        贵在坚持，加油！</p>

	</ul>    
</div>
