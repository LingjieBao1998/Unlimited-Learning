# SVG(Scalable Vector Graphics)

## 📖 目录
- [SVG(Scalable Vector Graphics)](#svgscalable-vector-graphics)
  - [📖 目录](#-目录)
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
  - [样式](#样式)
  - [reference](#reference)

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

去除填充——`fill`

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
| M    | x y  | 将画笔移动到点 (x, y) |
| L    | x y  | 画笔从当前的点绘制线段到点 (x, y) |
| H    | x    | 画笔从当前的点绘制水平线段到点 (x, y₀) |
| V    | y    | 画笔从当前的点绘制竖直线段到点 (x₀, y) |
| A    | rx ry x-axis-rotation large-arc-flag sweep-flag x y | 画笔从当前的点绘制一段圆弧到点 (x, y) |
| C    | x1 y1, x2 y2, x y | 画笔从当前的点绘制一段三次贝塞尔曲线到点 (x, y) |
| S    | x2 y2, x y | 特殊版本的三次贝塞尔曲线（省略第一个控制点） |
| Q    | x1 y1, x y | 绘制二次贝塞尔曲线到点 (x, y) |
| T    | x y  | 特殊版本的二次贝塞尔曲线（省略控制点） |
| Z    | 无参数 | 绘制闭合图形，如果 `d` 属性不指定 Z 命令，则绘制线段，而不是封闭图形。 |

## 样式



## reference
> http://www.aseoe.com/special/webstart/svg/ (强烈推荐) 