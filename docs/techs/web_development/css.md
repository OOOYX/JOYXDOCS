# CSS

---


##  关于CSS布局

> 默认的布局流是从上往下的

在CSS中使用**display, float, positon**这三个属性可以改变网页的布局流  

---

## display

### display:block

块级元素：占据一定矩形空间，可以设置width、height、margin、padding属性
> 例如，<div\> 默认属于块级元素

### display:inline

行内元素：依附于块级元素存在，因此width、height、margin、padding属性**不生效**
>例如，<a\> 默认属于行内元素

### display:flex

Flexbox：弹性盒子，用于创建横向或是纵向的一维页面布局

使用时，在想要进行Flex布局的元素的**父元素**上应用`display:flex`

```html
<div class="wrapper">
    <div>box1</div>
    <div>box2</div>
    <div>box3</div>
</div>
```

```css
.wrapper{
    display:flex
     
}
.wrapper > div{
    background-color:rgb(162, 247, 72);
    padding:1em;
    border-radius: 15px;
}
```

<div style="display:flex;">
    <div style="padding-top:1em">效果：</div>
    <div style="background-color:rgb(162, 247, 72);padding:1em;border-radius:15px;">box1</div>
    <div style="background-color:rgb(162, 247, 72);padding:1em;border-radius:15px;">box2</div>
    <div style="background-color:rgb(162, 247, 72);padding:1em;border-radius:15px;">box3</div>
</div>

#### *flex模型说明*

![img](https://developer.mozilla.org/zh-CN/docs/Learn/CSS/CSS_layout/Flexbox/flex_terms.png)

- **主轴**（main axis）是沿着 flex 元素放置的方向延伸的轴（比如页面上的横向的行、纵向的列）。该轴的开始和结束被称为 **main start** 和 **main end**。
- **交叉轴**（cross axis）是垂直于 flex 元素放置方向的轴。该轴的开始和结束被称为 **cross start** 和 **cross end**。
- 设置了 `display: flex` 的父元素被称之为 **flex 容器（flex container）。**
- 在 flex 容器中表现为弹性的盒子的元素被称之为 **flex 项**（**flex item**）

#### ***flex属性***

- **flex-direction**: 用于flex容器，指定主轴的方向，可用属性`column, row, column-reverse, row-reverse`
- **flex-wrap**:用于flex容器，指定换行规则，可用属性`wrap, nowrap, wrap-reverse`,  指定为wrap时，可以在在flex项上添加`flex`属性, 指定flex项在主轴上的最小尺寸
- **flex-flow**：对前两个属性的缩写表示，例如

```css
{
    flex-direction:row;
    flex-wrap:wrap;
}
```

可以简化为

```css
{
    flex-flow:row wrap;
}
```

- **flex**: 用于flex项，指定每个 flex 项沿主轴的可用空间大小。其规则为先分配带单位的尺寸，剩余空间按照不带单位的比例来分配。

例如

```css
.item1{
    flex:2 200px;
}
.item2{
    flex:1 200px;
}
```

表示每个 flex 项将首先给出 200px 的可用空间，然后，剩余的可用空间将根据分配的比例共享

