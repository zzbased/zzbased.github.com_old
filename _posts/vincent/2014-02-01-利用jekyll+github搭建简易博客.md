---
layout: post
category : jekyll
tagline: "Supporting tagline"
tags : [jekyll, blog]
---
{% include JB/setup %}

##利用jekyll+github搭建简易博客##

### Jekyll步骤

[Jekyll theme](http://jekyllthemes.org)

[Using Jekyll with Pages](https://help.github.com/articles/using-jekyll-with-pages/)

[在Github上搭建Jekyll博客和创建主题](http://yansu.org/2014/02/12/how-to-deploy-a-blog-on-github-by-jekyll.html)

[Jekyll中使用MathJax](http://www.pkuwwt.tk/linux/2013-12-03-jekyll-using-mathjax/)

- 在github上创建username.github.com目录。
- 安装jekyll，然后jekyll new username.github.com，将这些内容git push到github。

### Jekyll-bootstrap步骤 ###
1. Github工作目录：
搭建Github工作目录，需要先把ssh通道建立好，参看下面两篇文章。[产生ssh keys](https://help.github.com/articles/generating-ssh-keys), [可能碰到的问题](https://help.github.com/articles/error-permission-denied-publickey)

2. markdown编辑器：
在macbook上，我使用的编辑器是lightpaper. 引用图像存储链接服务是 [droplr](droplr.com)

3. 我使用的是[jekyllbootstrap](http://jekyllbootstrap.com)。号称三分钟可以教会搭建github博客，事实就是如此。参考这篇入门指南即可。[入门指南](http://jekyllbootstrap.com/usage/jekyll-quick-start.html)

	如果不喜欢默认的主题，可以按照示例做修改[jekyllthemes](http://jekyllthemes.org)，[jekyll-theming](http://jekyllbootstrap.com/usage/jekyll-theming.html)。

4. 需要注意的是，如果在上面准备工作里github的ssh设置没能成功。
	git remote set-url origin git@github.com:zzbased/zzbased.github.com.git
	可以更改为https地址:
	git remote set-url origin https://github.com/zzbased/zzbased.github.com.git

5. 安装好jekyll后，就可以本地调试。我们利用index.md，可以在原基础上做修改即可。

6. 然后在_post文件夹里，删除原来的example。利用rake post title="xxx"新增一个md文件。接下来就开始编辑了。

7. 如果不喜欢页面最下面的footer, 可以在“./_includes/themes/twitter/default.html”文件中，把footer屏蔽掉。不过建议还是留着，可以让更多的人接触到这项工具。

8. 在本地执行 jekyll serve,然后就可以在本机浏览器上通过0.0.0.0:4000预览网站。


备注：
目前我的博客使用的是原生的jekyll，没有采用jekyll-bootstrap。后来想想还是应该换用jekyll-bootstrap，因为它有评注的功能。

[常见的jekyll配置](http://jekyllbootstrap.com/usage/blog-configuration.html)

[关掉Jekyll Bootstrap的comments广告](http://stackoverflow.com/questions/19577049/jekyll-bootstrap-commenting-function-without-advertisement)
