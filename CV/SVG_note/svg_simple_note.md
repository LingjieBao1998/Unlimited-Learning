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
  - [reference](#reference)

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
* 
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
添加背景——`rect`

<svg width="160" height="160" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="white" />
<circle xmlns="http://www.w3.org/2000/svg" cx="80" cy="80" r="78" stroke-dasharray="5 5" stroke="black" fill="none" stroke-width="2"/>
</svg>

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
```svg
<svg width="300" height="200">
<text x="150" y="125" font-size="60" text-anchor="middle">SVG</text>
<svg>
```

<svg width="300" height="200">
<text x="150" y="125" font-size="60" text-anchor="middle">SVG</text>
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


## reference
> http://www.aseoe.com/special/webstart/svg/ (强烈推荐) 