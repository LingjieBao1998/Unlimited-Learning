# SVG(Scalable Vector Graphics)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [SVG(Scalable Vector Graphics)](#svgscalable-vector-graphics)
  - [introducation](#introducation)
    - [SVG与其它图片格式的比较](#svg与其它图片格式的比较)
  - [元素](#元素)
    - [The `<svg>` Element](#the-svg-element)
    - [circle](#circle)
    - [Rect——Rectangle](#rectrectangle)
    - [ellipse](#ellipse)
    - [line](#line)
    - [polyline](#polyline)
    - [polygon](#polygon)
    - [path(最通用)](#path最通用)
    - [text](#text)
  - [样式](#样式)
    - [fill——填充颜色](#fill填充颜色)
    - [stroke——边缘](#stroke边缘)
    - [rotate](#rotate)
  - [SVG重用与引用](#svg重用与引用)
    - [](#)
  - [svg与python](#svg与python)
    - [解析——lxml](#解析lxml)
    - [修改——lxml](#修改lxml)
    - [保存为图片/pdf——cairosvg](#保存为图片pdfcairosvg)
    - [写入——svgwrite](#写入svgwrite)
    - [text2svg](#text2svg)
  - [其他](#其他)
    - [rdkit显示svg](#rdkit显示svg)
  - [reference](#reference)
  - [svg 免费下载的网址](#svg-免费下载的网址)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## introducation
* SVG is used to define **vector-based graphics** for the Web
* SVG defines graphics in **XML** format
* SVG integrates with other standards, such as CSS, DOM, XSL and JavaScript
  
可缩放矢量图形(Scalable Vector Graphics，简称SVG)是一种使用**XML**来描述二维图形的语言(SVG严格遵从XML语法)。 SVG允许三种类型的图形对象：**矢量图形形状**（例如由直线和曲线组成的路径）、**图像**和**文本**。 可以将图形对象（包括文本）分组、样式化、转换和组合到以前呈现的对象中。 SVG 功能集包括嵌套转换、剪切路径、alpha 蒙板和模板对象。

**SVG既可以说是一种协议，也可以说是一门语言；既是HTML的一个标准元素，也是一种图片格式。**

### SVG与其它图片格式的比较
SVG与其它的图片格式相比，有很多优点(很多优点来源于矢量图的优点)：

* SVG文件是纯粹的XML， 可被非常多的工具读取和修改(比如记事本)。
* SVG 与JPEG 和GIF图像比起来，尺寸更小，且可压缩性更强。
* SVG 是可伸缩的，可在图像质量不下降的情况下被放大，可在任何的分辨率下被高质量地打印。
* SVG 图像中的文本是可选的，同时也是可搜索的(很适合制作地图)。

简单的例子
```svg
<svg width="200" height="200">
  <!-- face -->
  <circle cx="100" cy="100" r="90" fill="#39F" />
  <!-- eye -->
  <circle cx="70" cy="80" r="20" fill="white" />
  <circle cx="130" cy="80" r="20" fill="white" />
  <circle cx="65" cy="75" r="10" fill="black" />
  <circle cx="125" cy="75" r="10" fill="black" />
  <!-- smiles -->
  <path d="M 50 140 A 60 60 0 0 0 150 140" stroke="white" stoke-width="3" fill='none'>  
</svg>
```

<svg width="200" height="200">
  <!-- face -->
  <circle cx="100" cy="100" r="90" fill="#39F" />
  <!-- eye -->
  <circle cx="70" cy="80" r="20" fill="white" />
  <circle cx="130" cy="80" r="20" fill="white" />
  <circle cx="65" cy="75" r="10" fill="black" />
  <circle cx="125" cy="75" r="10" fill="black" />
  <!-- smiles -->
  <path d="M 50 140 A 60 60 0 0 0 150 140" stroke="white" stoke-width="3" fill='none'>  
</svg>

## 元素

### The `<svg>` Element
The HTML `<svg>` element is a container for SVG graphics.


### circle
这个元素的属性很简单，主要是定义圆心和半径：
* `r`：圆的半径。
* `cx`：圆心坐标x值。
* `cy`：圆心坐标y值。

```svg
<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
</svg>
```
<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
</svg>

* 虚线选项——stroke-dasharray="5 5"

去除填充——`fill="none"`
```svg
<svg width="200" height="200">
  <!-- 虚线圆，线段=5，间隔=3 -->
  <circle cx="100" cy="100" r="100" stroke="green" stroke-width="4" fill="none" stroke-dasharray="5 5" stroke-dashoffset="10" />
</svg>
```
<svg width="200" height="200">
  <!-- 虚线圆，线段=5，间隔=3 -->
  <circle cx="100" cy="100" r="100" stroke="green" stroke-width="4" fill="none" stroke-dasharray="5 5" stroke-dashoffset="10" />
</svg>


### Rect——Rectangle
* `x`：矩形左上角的坐标(用户坐标系)的x值。
* `y`：矩形左上角的坐标(用户坐标系)的y值。
* `width`：矩形宽度。
* `height`：矩形高度。
* `rx`：实现圆角效果时，圆角沿x轴的半径。
* `ry`：实现圆角效果时，圆角沿y轴的半径。

```svg
<svg width="400" height="120">
  <rect x="10" y="10" width="200" height="100" stroke="red" stroke-width="6" fill="blue" />
</svg>
```
<svg width="400" height="120">
  <rect x="10" y="10" width="200" height="100" stroke="red" stroke-width="6" fill="blue" />
</svg>

ref:https://www.w3schools.com/HTML/html5_svg.asp

可以用于添加背景（因为背景通常为**矩形**）

### ellipse
* `rx`：半长轴(x半径)。
* `ry`：半短轴(y半径)。
* `cx`：圆心坐标x值。
* `cy`：圆心坐标y值。

```svg
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="500" height="500">
  <ellipse cx="300" cy="80" rx="100" ry="50"
  style="fill:yellow;stroke:purple;stroke-width:2"/>
</svg>
```
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="500" height="150">
  <ellipse cx="300" cy="80" rx="100" ry="50"
  style="fill:yellow;stroke:purple;stroke-width:2"/>
</svg>

```svg
<svg xmlns="http://www.w3.org/2000/svg" version="1.1"  width="500" height="150">
  <ellipse cx="240" cy="50" rx="220" ry="30" style="fill:yellow"/>
  <ellipse cx="220" cy="50" rx="190" ry="20" style="fill:white"/>
</svg>
```

<svg xmlns="http://www.w3.org/2000/svg" version="1.1"  width="500" height="150">
  <ellipse cx="240" cy="50" rx="220" ry="30" style="fill:yellow"/>
  <ellipse cx="220" cy="50" rx="190" ry="20" style="fill:white"/>
</svg>

### line
* `x1`：起点x坐标。
* `y1`：起点y坐标。
* `x2`：终点x坐标。
* `y2`：终点y坐标。

```svg
<svg width="400" height="120">
  <line stroke="black" stroke-dasharray="5,5" stroke-width="2" x1="10" x2="200" y1="10" y2="100" />
</svg>
```
<svg width="400" height="120">
  <line stroke="black" stroke-dasharray="5,5" stroke-width="2" x1="10" x2="200" y1="10" y2="100" />
</svg>

### polyline
折线主要是要定义每条线段的端点即可，所以只需要一个点的集合作为参数：

points：一系列的用空格，逗号，换行符等分隔开的点。每个点必须有2个数字：x值和y值。所以下面3个点 (0,0), (1,1)和(2,2)可以写成："0 0, 1 1, 2 2"。

```svg
<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
<polyline points="50,50 100,150 150,100 200,200" fill="none" stroke="black" stroke-width="2" stroke-dasharray="5,5" />
</svg>
```

<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
<polyline points="50,50 100,150 150,100 200,200" fill="none" stroke="black" stroke-width="2" stroke-dasharray="5,5" />
</svg>

### polygon
这个元素就是比polyline元素多做一步，把最后一个点和第一个点连起来，形成闭合图形。参数是一样的。

points：一系列的用空格，逗号，换行符等分隔开的点。每个点必须有2个数字：x值和y值。所以下面3个点 (0,0), (1,1)和(2,2)可以写成："0 0, 1 1, 2 2"。

<svg viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
  <!-- 具有默认填充的多边形示例 -->
  <polygon points="0,100 50,25 50,75 100,0" />
</svg>

<svg viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
  <!-- 具有默认填充的多边形示例 -->
  <polygon points="0,100 50,25 50,75 100,0"  fill="none" stroke="black"  />
</svg>


### path(最通用)
| 指令 | 参数 | 说明 |
|------|------|------|
| M (**m**oveto)    | x y  | 将画笔移动到点 (x, y) |
| L (**l**ineto)   | x y  | 画笔从当前的点绘制线段到点 (x, y) |
| H (**h**orizontal lineto)   | x    | 画笔从当前的点绘制水平线段到点 (x, y₀) |
| V (**v**ertical lineto)   | y    | 画笔从当前的点绘制竖直线段到点 (x₀, y) |
| A (**e**lliptical Arc)   | rx ry x-axis-rotation large-arc-flag sweep-flag x y | 画笔从当前的点绘制一段圆弧到点 (x, y) |
| C (**c**urveto)   | x1 y1, x2 y2, x y | 画笔从当前的点绘制一段三次贝塞尔曲线到点 (x, y) |
| S (**s**mooth curveto)   | x2 y2, x y | 特殊版本的三次贝塞尔曲线（省略第一个控制点） |
| Q (**q**uadratic Bézier curve)   | x1 y1, x y | 绘制二次贝塞尔曲线到点 (x, y) |
| T (smooth quadratic Bézier curveto)   | x y  | 特殊版本的二次贝塞尔曲线（省略控制点） |
| Z (closepath)   | 无参数 | 绘制闭合图形，如果 `d` 属性不指定 Z 命令，则绘制线段，而不是封闭图形。 |

> 这个属性相对比较复杂，后续有空再补充


### text
* `x` 和 `y` 属性定义了文本左上角的坐标，即文本的起始点位置。
* `font-family` 属性定义了文本的字体名称，可以是系统字体或自定义字体。
* `font-size` 属性定义了文本的字体大小，以像素为单位。
* `fill` 属性定义了文本的颜色。
* `text-anchor` 属性定义了文本锚点，即文本相对于指定坐标的对齐方式，常用取值有 `"start"`（默认，左对齐）、`"middle"`（居中对齐）和 `"end"`（右对齐）。

> `<text>`:通常无法给出准确的`bbox`位置
```svg
<svg width="300" height="200">
<text x="150" y="125" font-size="60" text-anchor="middle">SVG</text>
<svg>
```

<svg width="300" height="200">
<text x="150" y="125" font-size="60" text-anchor="middle">SVG</text>
<svg>

```svg
<svg width="300" height="200">
<text x="150" y="125" font-size="60" text-anchor="middle">()</text>
<svg>
```

<svg width="300" height="200">
<text x="150" y="125" font-size="60" text-anchor="middle">()</text>
<svg>

```svg
<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <text x="100" y="100" font-family="Arial" font-size="20" fill="blue" text-anchor="middle">Hello, SVG!</text>
</svg>
```
<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <text x="100" y="100" font-family="Arial" font-size="20" fill="blue" text-anchor="middle">Hello, SVG!</text>
</svg>

```
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="0" y="15" fill="red" transform="rotate(30 20,40)">I love SVG</text>
</svg>
```
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="0" y="15" fill="red" transform="rotate(30 20,40)">I love SVG</text>
</svg>

> transform="rotate(30 20,40)"
这个 transform 属性是 SVG 中用来对元素进行几何变换（如旋转、平移、缩放等）的属性。具体到 rotate(30 20,40)，它表示对 <text> 元素应用一个​​旋转变换​​。这个 rotate() 函数的语法是：
```svg
rotate(<angle> [<cx> <cy>])
```
* `<angle>`：旋转的角度，单位是​​度（degrees）​​，正值表示​​逆时针​​旋转，负值表示​​顺时针​​旋转。
* `<cx>` `<cy>`（可选）：旋转中心的坐标（x, y），如果不指定，默认以元素的​​原点（通常是左上角）​​为旋转中心。



## 样式
### fill——填充颜色
* 如果不提供`fill`属性，则默认会使用黑色填充,如果要取消填充，需要设置成`"none"`。
* 可以设置填充的透明度，就是fill-opacity，值的范围是0到1。

### stroke——边缘
* 如果不提供`stroke`属性，则默认不绘制图形边框。
* 可以设置边的透明度，就是`stroke-opacity`，值的范围是0到1。
* `stroke-width`定义线段的宽度
* `stroke-linecap`边缘的形状
```svg
<svg xmlns="https://www.w3.org/2000/svg" version="1.1">
  <g fill="none" stroke="black" stroke-width="6">
    <path stroke-linecap="butt" d="M5 20 l215 0" />
    <path stroke-linecap="round" d="M5 40 l215 0" />
    <path stroke-linecap="square" d="M5 60 l215 0" />
  </g>
</svg>
```
<svg xmlns="https://www.w3.org/2000/svg" version="1.1">
  <g fill="none" stroke="black" stroke-width="6">
    <path stroke-linecap="butt" d="M5 20 l215 0" />
    <path stroke-linecap="round" d="M5 40 l215 0" />
    <path stroke-linecap="square" d="M5 60 l215 0" />
  </g>
</svg>

* `stroke-dasharray`——边缘的样式
```svg
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <g fill="none" stroke="black" stroke-width="4">
    <path stroke-dasharray="5,5" d="M5 20 l215 0" />
    <path stroke-dasharray="10,10" d="M5 40 l215 0" />
    <path stroke-dasharray="20,10,5,5,5,10" d="M5 60 l215 0" />
  </g>
</svg>
```
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <g fill="none" stroke="black" stroke-width="4">
    <path stroke-dasharray="5,5" d="M5 20 l215 0" />
    <path stroke-dasharray="10,10" d="M5 40 l215 0" />
    <path stroke-dasharray="20,10,5,5,5,10" d="M5 60 l215 0" />
  </g>
</svg>

### rotate
```svg
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="100" y="15" fill="#666666" >Hello SVG</text>
</svg>
```

<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="100" y="15" fill="#666666" >Hello SVG</text>
</svg>


```svg
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="100" y="15" fill="#666666" rotate="46 10,10" >Hello SVG</text>
</svg>
```

<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="100" y="15" fill="#666666" rotate="46 10,10" >Hello SVG</text>
</svg>




## SVG重用与引用
### <g>
g元素是一种容器，它组合一组相关的图形元素成为一个整体

<svg width="1144.12px" height="400px" viewBox="0 0 572.06 200">
    <style>
        svg{background-color:white;}
        #wing{fill:#81CCAA;}
        #body{fill:#B8E4C2;}
        #pupil{fill:#1F2600;}
        #beak{fill:#F69C0D;}
        .eye-ball{fill:#F6FDC4;}
    </style>
    <g id="bird">
        <g id="body">
            <path d="M48.42,78.11c0-17.45,14.14-31.58,31.59-31.58s31.59,14.14,31.59,31.58c0,17.44-14.14,31.59-31.59,31.59
            S48.42,95.56,48.42,78.11"/>
            <path d="M109.19,69.88c0,0-8.5-27.33-42.51-18.53c-34.02,8.81-20.65,91.11,45.25,84.73
            c40.39-3.65,48.59-24.6,48.59-24.6S124.68,106.02,109.19,69.88"/>
            <path id="wing" d="M105.78,75.09c4.56,0,8.84,1.13,12.62,3.11c0,0,0.01-0.01,0.01-0.01l36.23,12.38c0,0-13.78,30.81-41.96,38.09
            c-1.51,0.39-2.82,0.59-3.99,0.62c-0.96,0.1-1.92,0.16-2.9,0.16c-15.01,0-27.17-12.17-27.17-27.17
            C78.61,87.26,90.78,75.09,105.78,75.09"/>
        </g>
        <g id="head">
            <path id="beak" d="M50.43,68.52c0,0-8.81,2.58-10.93,4.86l9.12,9.87C48.61,83.24,48.76,74.28,50.43,68.52"/>
            <path class="eye-ball" d="M60.53,71.68c0-6.33,5.13-11.46,11.46-11.46c6.33,0,11.46,5.13,11.46,11.46c0,6.33-5.13,11.46-11.46,11.46
                C65.66,83.14,60.53,78.01,60.53,71.68"/>
            <path id="pupil" d="M64.45,71.68c0-4.16,3.38-7.53,7.54-7.53c4.16,0,7.53,3.37,7.53,7.53c0,4.16-3.37,7.53-7.53,7.53
                C67.82,79.22,64.45,75.84,64.45,71.68"/>
            <path class="eye-ball" d="M72.39,74.39c0-2.73,2.22-4.95,4.95-4.95c2.73,0,4.95,2.21,4.95,4.95c0,2.74-2.22,4.95-4.95,4.95
                C74.6,79.34,72.39,77.13,72.39,74.39"/>
        </g>
    </g>
</svg>

## svg与python
### 解析——lxml
```python
from lxml import etree

#读取SVG文件
with open('example.svg', 'r') as file:
  svg_content = file.read()

# 解析SVG内容
svg_root = etree.fromstring(svg_content)

# 获取SVG文件的根节点
print(svg_root.tag)  # 输出: {http://www.w3.org/2000/svg}svg

# 获取SVG文件中的所有元素
for element in svg_root.iter():
  print(element.tag, element.attrib)
```
> 另外`svglib`也可以读取svg元素

### 修改——lxml
附上
```python
paths = svg_root.findall('.//{http://www.w3.org/2000/svg}path')


for path in paths:
  print(path.attrib['d'])  # 输出路径的d属性


# 修改某个路径的d属性
if paths:
  paths[0].attrib['d'] = 'M10 10 H 90 V 90 H 10 L 10 10'
```
> 另外`svglib`也可以修改svg元素

### 保存为图片/pdf——cairosvg
```python
import cairosvg
# 渲染SVG文件为PNG格式
cairosvg.svg2png(url='example.svg', write_to='output.png')

# 渲染SVG文件为PDF格式
cairosvg.svg2pdf(url='example.svg', write_to='output.pdf')
```

### 写入——svgwrite
```python
import svgwrite

# 创建一个新的SVG文件
dwg = svgwrite.Drawing('output.svg', profile='tiny')


# 添加一个圆形元素
dwg.add(dwg.circle(center=(50, 50), r=40, fill='red'))


# 保存SVG文件
dwg.save()


# 读取并操作现有的SVG文件
with open('example.svg', 'r') as file:
  svg_content = file.read()
  print(svg_content)
```

ref:https://github.com/janily/rocksvg/blob/master/docs/01_svg%E5%9F%BA%E6%9C%AC%E7%9F%A5%E8%AF%86/07_%E5%85%83%E7%B4%A0%E7%9A%84%E9%87%8D%E7%94%A8%E4%B8%8E%E5%BC%95%E7%94%A8.md

### text2svg
有时候一些特殊字符无法被系统默认的字体所显示，这就需要先将其转化成`path`对象，而非`text`对象进行显示
```python
#pip install text2svg
from text2svg import *
info = TextInfo("Hello World","hello.svg",50,50)
text2svg(info)
```

<a href="./hello.svg">hello.svg</a>
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="50pt" height="50pt" viewBox="0 0 50 50" version="1.1">
<defs>
<g>
<symbol overflow="visible" id="glyph0-0">
<path style="stroke:none;" d="M 1.25 0 L 1.25 -9.515625 L 6.734375 -9.515625 L 6.734375 0 L 1.25 0 Z M 1.9375 -0.6875 L 6.046875 -0.6875 L 6.046875 -8.84375 L 1.9375 -8.84375 L 1.9375 -0.6875 Z M 1.9375 -0.6875 "/>
</symbol>
<symbol overflow="visible" id="glyph0-1">
<path style="stroke:none;" d="M 8.5625 0 L 7.375 0 L 7.375 -4.4375 L 2.5 -4.4375 L 2.5 0 L 1.296875 0 L 1.296875 -9.515625 L 2.5 -9.515625 L 2.5 -5.484375 L 7.375 -5.484375 L 7.375 -9.515625 L 8.5625 -9.515625 L 8.5625 0 Z M 8.5625 0 "/>
</symbol>
<symbol overflow="visible" id="glyph0-2">
<path style="stroke:none;" d="M 3.890625 -7.28125 C 4.503906 -7.28125 5.03125 -7.144531 5.46875 -6.875 C 5.90625 -6.613281 6.242188 -6.238281 6.484375 -5.75 C 6.722656 -5.269531 6.84375 -4.703125 6.84375 -4.046875 L 6.84375 -3.34375 L 1.953125 -3.34375 C 1.960938 -2.53125 2.164062 -1.910156 2.5625 -1.484375 C 2.957031 -1.066406 3.507812 -0.859375 4.21875 -0.859375 C 4.675781 -0.859375 5.078125 -0.898438 5.421875 -0.984375 C 5.773438 -1.078125 6.140625 -1.203125 6.515625 -1.359375 L 6.515625 -0.328125 C 6.148438 -0.171875 5.789062 -0.0546875 5.4375 0.015625 C 5.082031 0.0976562 4.660156 0.140625 4.171875 0.140625 C 3.492188 0.140625 2.894531 0.00390625 2.375 -0.265625 C 1.851562 -0.546875 1.445312 -0.957031 1.15625 -1.5 C 0.875 -2.050781 0.734375 -2.722656 0.734375 -3.515625 C 0.734375 -4.296875 0.863281 -4.96875 1.125 -5.53125 C 1.382812 -6.09375 1.753906 -6.523438 2.234375 -6.828125 C 2.710938 -7.128906 3.265625 -7.28125 3.890625 -7.28125 Z M 3.875 -6.3125 C 3.3125 -6.3125 2.867188 -6.128906 2.546875 -5.765625 C 2.222656 -5.410156 2.03125 -4.914062 1.96875 -4.28125 L 5.609375 -4.28125 C 5.597656 -4.882812 5.453125 -5.375 5.171875 -5.75 C 4.898438 -6.125 4.46875 -6.3125 3.875 -6.3125 Z M 3.875 -6.3125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-3">
<path style="stroke:none;" d="M 2.3125 0 L 1.140625 0 L 1.140625 -10.125 L 2.3125 -10.125 L 2.3125 0 Z M 2.3125 0 "/>
</symbol>
<symbol overflow="visible" id="glyph0-4">
<path style="stroke:none;" d="M 7.34375 -3.578125 C 7.34375 -2.398438 7.039062 -1.484375 6.4375 -0.828125 C 5.84375 -0.179688 5.035156 0.140625 4.015625 0.140625 C 3.378906 0.140625 2.816406 -0.00390625 2.328125 -0.296875 C 1.835938 -0.585938 1.445312 -1.007812 1.15625 -1.5625 C 0.875 -2.125 0.734375 -2.796875 0.734375 -3.578125 C 0.734375 -4.765625 1.03125 -5.675781 1.625 -6.3125 C 2.21875 -6.957031 3.023438 -7.28125 4.046875 -7.28125 C 4.703125 -7.28125 5.273438 -7.132812 5.765625 -6.84375 C 6.253906 -6.550781 6.640625 -6.128906 6.921875 -5.578125 C 7.203125 -5.035156 7.34375 -4.367188 7.34375 -3.578125 Z M 1.953125 -3.578125 C 1.953125 -2.734375 2.117188 -2.066406 2.453125 -1.578125 C 2.785156 -1.085938 3.3125 -0.84375 4.03125 -0.84375 C 4.757812 -0.84375 5.289062 -1.085938 5.625 -1.578125 C 5.957031 -2.066406 6.125 -2.734375 6.125 -3.578125 C 6.125 -4.421875 5.957031 -5.082031 5.625 -5.5625 C 5.289062 -6.050781 4.757812 -6.296875 4.03125 -6.296875 C 3.300781 -6.296875 2.769531 -6.050781 2.4375 -5.5625 C 2.113281 -5.082031 1.953125 -4.421875 1.953125 -3.578125 Z M 1.953125 -3.578125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-5">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="glyph0-6">
<path style="stroke:none;" d="M 12.21875 -9.515625 L 9.6875 0 L 8.484375 0 L 6.625 -6.234375 C 6.550781 -6.492188 6.476562 -6.75 6.40625 -7 C 6.332031 -7.257812 6.273438 -7.488281 6.234375 -7.6875 C 6.191406 -7.894531 6.160156 -8.035156 6.140625 -8.109375 C 6.128906 -7.992188 6.082031 -7.75 6 -7.375 C 5.914062 -7 5.8125 -6.609375 5.6875 -6.203125 L 3.890625 0 L 2.671875 0 L 0.15625 -9.515625 L 1.40625 -9.515625 L 2.890625 -3.703125 C 2.992188 -3.296875 3.082031 -2.898438 3.15625 -2.515625 C 3.238281 -2.140625 3.300781 -1.78125 3.34375 -1.4375 C 3.382812 -1.78125 3.445312 -2.15625 3.53125 -2.5625 C 3.625 -2.976562 3.734375 -3.378906 3.859375 -3.765625 L 5.53125 -9.515625 L 6.765625 -9.515625 L 8.515625 -3.734375 C 8.640625 -3.328125 8.742188 -2.921875 8.828125 -2.515625 C 8.921875 -2.109375 8.988281 -1.75 9.03125 -1.4375 C 9.082031 -1.769531 9.144531 -2.128906 9.21875 -2.515625 C 9.300781 -2.898438 9.394531 -3.300781 9.5 -3.71875 L 10.96875 -9.515625 L 12.21875 -9.515625 Z M 12.21875 -9.515625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-7">
<path style="stroke:none;" d="M 4.46875 -7.28125 C 4.601562 -7.28125 4.742188 -7.269531 4.890625 -7.25 C 5.046875 -7.238281 5.179688 -7.222656 5.296875 -7.203125 L 5.15625 -6.125 C 5.039062 -6.144531 4.914062 -6.160156 4.78125 -6.171875 C 4.644531 -6.191406 4.515625 -6.203125 4.390625 -6.203125 C 4.023438 -6.203125 3.679688 -6.101562 3.359375 -5.90625 C 3.035156 -5.707031 2.78125 -5.425781 2.59375 -5.0625 C 2.40625 -4.707031 2.3125 -4.289062 2.3125 -3.8125 L 2.3125 0 L 1.140625 0 L 1.140625 -7.140625 L 2.09375 -7.140625 L 2.21875 -5.84375 L 2.28125 -5.84375 C 2.507812 -6.226562 2.804688 -6.5625 3.171875 -6.84375 C 3.535156 -7.132812 3.96875 -7.28125 4.46875 -7.28125 Z M 4.46875 -7.28125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-8">
<path style="stroke:none;" d="M 3.671875 0.140625 C 2.773438 0.140625 2.0625 -0.164062 1.53125 -0.78125 C 1 -1.40625 0.734375 -2.332031 0.734375 -3.5625 C 0.734375 -4.78125 1 -5.703125 1.53125 -6.328125 C 2.070312 -6.960938 2.785156 -7.28125 3.671875 -7.28125 C 4.222656 -7.28125 4.675781 -7.175781 5.03125 -6.96875 C 5.382812 -6.757812 5.671875 -6.507812 5.890625 -6.21875 L 5.96875 -6.21875 C 5.957031 -6.332031 5.941406 -6.503906 5.921875 -6.734375 C 5.898438 -6.960938 5.890625 -7.144531 5.890625 -7.28125 L 5.890625 -10.125 L 7.0625 -10.125 L 7.0625 0 L 6.125 0 L 5.9375 -0.953125 L 5.890625 -0.953125 C 5.679688 -0.648438 5.394531 -0.390625 5.03125 -0.171875 C 4.675781 0.0351562 4.222656 0.140625 3.671875 0.140625 Z M 3.859375 -0.84375 C 4.609375 -0.84375 5.132812 -1.046875 5.4375 -1.453125 C 5.75 -1.867188 5.90625 -2.492188 5.90625 -3.328125 L 5.90625 -3.546875 C 5.90625 -4.429688 5.757812 -5.109375 5.46875 -5.578125 C 5.175781 -6.054688 4.632812 -6.296875 3.84375 -6.296875 C 3.207031 -6.296875 2.734375 -6.046875 2.421875 -5.546875 C 2.109375 -5.046875 1.953125 -4.375 1.953125 -3.53125 C 1.953125 -2.675781 2.109375 -2.015625 2.421875 -1.546875 C 2.734375 -1.078125 3.210938 -0.84375 3.859375 -0.84375 Z M 3.859375 -0.84375 "/>
</symbol>
</g>
</defs>
<g id="surface0">
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-1" x="0" y="15"/>
  <use xlink:href="#glyph0-2" x="10" y="15"/>
  <use xlink:href="#glyph0-3" x="18" y="15"/>
  <use xlink:href="#glyph0-3" x="21" y="15"/>
  <use xlink:href="#glyph0-4" x="24" y="15"/>
  <use xlink:href="#glyph0-5" x="32" y="15"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-6" x="0" y="34"/>
  <use xlink:href="#glyph0-4" x="12" y="34"/>
  <use xlink:href="#glyph0-7" x="20" y="34"/>
  <use xlink:href="#glyph0-3" x="26" y="34"/>
  <use xlink:href="#glyph0-8" x="29" y="34"/>
</g>
</g>
</svg>

> ref:https://pypi.org/project/text2svg/


## 其他
### rdkit显示svg
```svg
<svg baseProfile="full" height="200px" version="1.1" width="600px" xml:space="preserve" xmlns:rdkit="http://www.rdkit.org/xml" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g transform="translate(0,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#000000" x="84.4958" y="108.25"><tspan>CH</tspan><tspan style="baseline-shift:sub;font-size:11.25px;">4</tspan><tspan/></text>
</g>
<g transform="translate(200,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<path d="M 9.09091,100 59.1479,100" style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<path d="M 59.1479,100 109.205,100" style="fill:none;fill-rule:evenodd;stroke:#FF0000;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#FF0000" x="109.205" y="107.5"><tspan>OH</tspan></text>
</g>
<g transform="translate(400,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<path d="M 9.09091,100 55.1606,100" style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<path d="M 55.1606,100 101.23,100" style="fill:none;fill-rule:evenodd;stroke:#0000FF;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#0000FF" x="101.23" y="108.25"><tspan>NH</tspan><tspan style="baseline-shift:sub;font-size:11.25px;">2</tspan><tspan/></text>
</g></svg>
```

<svg baseProfile="full" height="200px" version="1.1" width="600px" xml:space="preserve" xmlns:rdkit="http://www.rdkit.org/xml" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g transform="translate(0,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#000000" x="84.4958" y="108.25"><tspan>CH</tspan><tspan style="baseline-shift:sub;font-size:11.25px;">4</tspan><tspan/></text>
</g>
<g transform="translate(200,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<path d="M 9.09091,100 59.1479,100" style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<path d="M 59.1479,100 109.205,100" style="fill:none;fill-rule:evenodd;stroke:#FF0000;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#FF0000" x="109.205" y="107.5"><tspan>OH</tspan></text>
</g>
<g transform="translate(400,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<path d="M 9.09091,100 55.1606,100" style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<path d="M 55.1606,100 101.23,100" style="fill:none;fill-rule:evenodd;stroke:#0000FF;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"/>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#0000FF" x="101.23" y="108.25"><tspan>NH</tspan><tspan style="baseline-shift:sub;font-size:11.25px;">2</tspan><tspan/></text>
</g></svg>

> ref:https://github.com/rdkit/rdkit-tutorials/issues/5

其他rdkit的样式
上标 & 下标 & 特殊字符

<svg baseProfile="full" height="200px" version="1.1" width="600px" xml:space="preserve" xmlns:rdkit="http://www.rdkit.org/xml" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g transform="translate(0,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#000000" x="84.4958" y="108.25"><tspan>CH</tspan><tspan style="baseline-shift:super;font-size:11.25px;">4</tspan><tspan/></text>
</g>
<g transform="translate(200,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:13.5px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#FF0000" x="109.23" y="93.25"><tspan>⊖</tspan></text> 
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#FF0000" x="109.205" y="107.5"><tspan>O</tspan><tspan style="baseline-shift:super;font-size:11.25px;">​⊖</tspan></text>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#FF0000" x="109.205" y="123.5"><tspan>H</tspan></text>

</g>
<g transform="translate(400,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:13.5px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#0000FF" x="101.23" y="93.25"><tspan>​​⨁</tspan></text> <!-- 加个字体高度就了 -->
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#0000FF" x="101.23" y="108.25"><tspan>N</tspan><tspan/><tspan style="baseline-shift:super;font-size:11.25px;">​​⨁</tspan></text>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#0000FF" x="101.23" y="123.25"><tspan>H</tspan><tspan style="baseline-shift:sub;font-size:11.25px;">2</tspan><tspan/></text> <!-- 加个字体高度就了 -->
</g></svg>



<svg baseProfile="full" height="200px" version="1.1" width="600px" xml:space="preserve" xmlns:rdkit="http://www.rdkit.org/xml" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g transform="translate(0,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#000000" x="84.4958" y="108.25"><tspan>CH</tspan><tspan style="baseline-shift:sub;font-size:11.25px;">4</tspan><tspan/></text>
</g>
<g transform="translate(200,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#FF0000" x="109.205" y="107.5"><tspan>O</tspan></text>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#FF0000" x="109.205" y="123.5"><tspan>TS</tspan></text>
</g>
<g transform="translate(400,0)"><rect height="200" style="opacity:1.0;fill:#FFFFFF;stroke:none" width="200" x="0" y="0"> </rect>
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#0000FF" x="101.23" y="93.25"><tspan>​​Boc</tspan></text> <!-- 加个字体高度就了 -->
<text style="font-size:15px;font-style:normal;font-weight:normal;fill-opacity:1;stroke:none;font-family:sans-serif;text-anchor:start;fill:#0000FF" x="101.23" y="108.25"><tspan>N</tspan><tspan/></text>
</g></svg>

                            


## reference
> http://www.aseoe.com/special/webstart/svg/ (强烈推荐) 

## svg 免费下载的网址
> https://www.svgrepo.com/
