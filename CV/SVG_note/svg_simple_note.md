# SVG(Scalable Vector Graphics)
* SVG is used to define **vector-based graphics** for the Web
* SVG defines graphics in **XML** format
* SVG integrates with other standards, such as CSS, DOM, XSL and JavaScript

## The `<svg>` Element
The HTML `<svg>` element is a container for SVG graphics.

## circle
```svg
<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
</svg>
```
<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
</svg>

## Rectangle
```svg
<svg width="400" height="120">
  <rect x="10" y="10" width="200" height="100" stroke="red" stroke-width="6" fill="blue" />
</svg>
```
<svg width="400" height="120">
  <rect x="10" y="10" width="200" height="100" stroke="red" stroke-width="6" fill="blue" />
</svg>

ref:https://www.w3schools.com/HTML/html5_svg.asp

## dash line
```svg
<svg width="400" height="120">
  <line stroke="black" stroke-dasharray="5,5" stroke-width="2" x1="10" x2="200" y1="10" y2="100" />
</svg>
```