---
title: About Markdown
author: east
date: 2022-09-13 00:00:00 +09:00
categories: [Open Source, Markdown]
tags: [Open Source, Markdown]
math: true
mermaid: true
---


> # Ⅰ. 서론

Markdown 이라는 것은 github를 사용하면서 README.md로 처음 접하게 되었다.

github를 사용하는데 있어 .md 파일은 프로젝트의 문서화와 gitblog를 시작하게 되면서 post의 형태도 .md의 형태로 작성하다 보니 관심이 동했다

또한, 궁극적으로 인터넷으로 매우 잘 나와있지만 한번에 보기 쉬운 cheat sheet가 없어서 직접 구성하게 되었다.

> # Ⅱ. 본론


## ⅰ. Markdown(마크다운)이란?

    > _**[Markdown][2] is lightweight [markup][3] language**_  

    > Different than using a [WYSIWYG(What You See Is What You Get)][1]  
    > &nbsp; $\because$ &nbsp; .md isn't  visible immediately.
    
    
## ⅱ. 마크다운을 쓰는 이유

    다양한 이유들이 있지만 가장 큰 장점은  
    `다양한 플랫폼이나 프로그램에서 사용이 가능하다는 점`일 것이다.(특히, Github, Reddit 등)
    

## ⅲ. Markdown Cheat Sheet

- Line Break
    Line Breaks : two or more space(줄바꿈 space 두 번)  
    Paragraphs : one or more lines(문단바꿈은 enter 두 번)  
    Space : 스페이스바를 많이 눌러도 한 칸으로 표현. (& + nbsp; 사용)


- Heading  
    ```markdown
    # Heading {#heading-id}
    ## Heading

    Heading
    =======
    Heading
    -------

    [Heading Ids](#heading-id)
    ```

- ### Emphasis 
    ```markdown
    __볼드체__, **볼드체**, <strong>
    _이텔릭체_, *이텔릭체*, <em>
    ```
    <strong>Bold</strong>
    __볼드체__, **볼드체**  

    <em>italic</em>
    _이텔릭체_, *이텔릭체*  

- ### Strikethrough & UnderLine
    ```markdown
    ~~Strikethrough~~
    <ins>Underline</ins>
    ```

    ~~Strikethrough~~  
    <ins>Underline</ins>

- ### blockquote
    ```markdown
    > blockquote
    >> blockquote
    > 
    > blockqote
    ```

    > blockquote
    >> blockquote
    > 
    > blockqote

- ### tabs(들여쓰기)
    ```markdown
        tabs
        tabs
    tabs
    ```
        tabs
        tabs  
    tabs            

- ### Ordered List
    ```markdown
    1. list
    1. list
    1. list
    ```  

    1. list
    1. list
    1. list

- ### Unordered List
    ```html
    - list
        text
        > qudote
    - list
        - list
            - list
    ```  
    - list
        text
        > quote
    - list
        - list
            - list

- ### Definition Lists
    ```markdown
    First Term
    : This is the definition of ther first term.
    ```

    First Term
    : This is the definition of ther first term.

- ### Image
    ```markdown
    ![iamge](image url)
    ![google!]('https://www.google.co.kr/images/branding/googlelogo/2x/googlelogo_color_160x56dp.png' "San Juan Mountains")
    [![Linking Image]('https://www.google.co.kr/images/branding/googlelogo/2x/googlelogo_color_160x56dp.png' "Shiprock, New Mexico by Beau Rogers")](url)
    ```
 
    ![google!](https://www.google.co.kr/images/branding/googlelogo/2x/googlelogo_color_160x56dp.png "San Juan Mountains")  
    [![Linking Image](https://www.google.co.kr/images/branding/googlelogo/2x/googlelogo_color_160x56dp.png "Shiprock, New Mexico by Beau Rogers")](https://www.google.co.kr/images/branding/googlelogo/2x/googlelogo_color_160x56dp.png)  
    
- ### Code Blocks
    ````markdown
    `inline code`

    ```Syntax Highlighting(python, json, ...)
    {
    "firstName": "John",
    "lastName": "Smith",
    "age": 25
    }

    ```
    ````

    `inline code`
    ```json
        {
        "firstName": "John",
        "lastName": "Smith",
        "age": 25
        }
    ```

    > <dl>
    > <dt>How to escape 3-backtick in code block?</dt>
    > <dd> : use more backtick than 3-backtick.</dd>
    > </dl>

- ### Horizontal Rules
    ```markdown
    *** 
    ---
    _______
    ```  
    *** 
    ---
    _______

- ### Link
    ```markdown
    Automatic Linking, not use anything
    https://en.wikipedia.org/wiki

    disableing Automatic Linking
    `https://en.wikipedia.org/wiki

    [click here](url)

    [wiki][1]  
    [1]: https://en.wikipedia.org/wiki
    [1]: https://en.wikipedia.org/wiki "wiki"
    ```
    https://en.wikipedia.org/wiki  
    `https://en.wikipedia.org/wiki  
    [click here](https://en.wikipedia.org/wiki)   
    
    [wiki][10]  
    
    [10]: https://en.wikipedia.org/wiki

- ### Escape
    ```markdown
    \* without the backslash, this would be a bullet in an unordered list.
    ```
    \* without the backslash, this would be a bullet in an unordered list.

- ### Definition
    ```markdown
    First Term # github에는 적용이 안됨. 그래서 html 형태로 작성해야 함.
    : This is the definition of ther first term.

    <dl>
    <dt>First Term</dt>
    <dd> : This is the definition of ther first term.</dd>
    </dl>
    ```

    <dl>
    <dt>First Term</dt>
    <dd> : This is the definition of ther first term.</dd>
    </dl>

- ### Task list

    ```markdown
    - [x] task1
    - [ ] task2
    ```  

    - [x] task1
    - [ ] task2

- ### Subscript
    ```markdown
    H~2~O ~로 감싸기
    H<sub>2</sub>O
    ```
    H~2~O  
    H<sub>2</sub>O

- ### Superscript
    ```markdown
    X^2^ ^로 감싸기
    X<sup>2</sup>
    ```
    X^2^  
    X<sup>2</sup>

- ### Footnote(각주)
    ```markdown
    footnote,[^1] and here's a longer one.[^bignote]

    [^1]: This is the first one.
    [^bignote]: longer one

    ```
    footnote,[^1] and here's a longer one.[^bignote]

    [^1]: This is the first one.
    [^bignote]: longer one

- ### Table
    ```markdown    
    |a | b| c|
    | :--- | :---: | ---: |
    |왼쪽 | 가운데 | 오른쪽 | 
    ```  
    |a | b| c|
    | :--- | :---: | ---: |
    |왼쪽 | 가운데 | 오른쪽 | 

- ### Highlight
    ```markdown
    ==highlight==

    <mark>highlight</mark>
    ```
    ==highlight==  
    <mark>highlight</mark>    

- ### not officially upported by Markdown
    - #### Center
        <center>center text.</center>

    - #### Color
        <font color="red">This text is red!</font>
        <p style="color:blue">Make this text blue.</p>

    - #### Comments
        ```markdown
        Here's a paragraph that will be visible.

        [This is a comment that will be hidden.]: # 

        And here's another paragraph that's visible.
        ```

        Here's a paragraph that will be visible.

        [This is a comment that will be hidden.]: # 

        And here's another paragraph that's visible.

    - #### Admonitions & emoji
        > :warning: **Warning:** Do not push the big red button.

        > :memo: **Note:** Sunrises are beautiful.

        > :bulb: **Tip:** Remember to appreciate the little things in life.

        :mag: [emoji cheat sheet][4]

    - #### Image caption
        <figure>
            <img src="https://www.google.co.kr/images/branding/googlelogo/2x/googlelogo_color_160x56dp.png"
                alt="Albuquerque, New Mexico">
            <figcaption>A single track trail outside of Albuquerque, New Mexico.</figcaption>
        </figure>

        ![Albuquerque, New Mexico](https://www.google.co.kr/images/branding/googlelogo/2x/googlelogo_color_160x56dp.png)  
        *A single track trail outside of Albuquerque, New Mexico.*

    - #### Link Targets(new tabs)
        <a href="https://www.markdownguide.org" target="_blank">Learn Markdown!</a>

    - #### Symbols
  
        |Symbols | |
        | --- | --- |
        |Copyright (©) | \&copy; |
        |Registered trademark (®) | \&reg;|
        |Trademark (™) | \&trade;|
        |Euro (€) | \&euro;|
        |Left arrow (←) | \&larr;|
        |Up arrow (↑) | \&uarr;|
        |Right arrow (→) | \&rarr;| 
        |Down arrow (↓) | \&darr;|
        |Degree (°) | \&#176;|
        |Pi (π) | \&#960;|

    - #### Mathematical expression(수식)

        ```markdown 
        # 정렬
        $ x + y = 1$    # 기본 정렬
        $$ x + y = 1 $$ # 중앙 정렬
        $$              # aligned 심볼을 통해 특정 문자 기준 정렬(&를 기준)
        \begin{aligned} 
        f(x)&=ax^2+bx+c\\
        g(x)&=Ax^4
        \end{aligned}$$
        ```
        
        $ x + y = 1$  
        $$ x + y = 1 $$  
        $$
        \begin{aligned}
        f(x)&=ax^2+bx+c\\
        g(x)&=Ax^4
        \end{aligned}$$  

        ```markdown
        # 줄바꿈
        $$x+y=3\\-x+3y=2$$  # enter로 줄바꿈이 안되고 \\ 를통해 줄바꿈  
        ```
        
        $$x+y=3\\-x+3y=2$$
        
        ```markdown
        # 띄어쓰기
        $local minimum$     # 적용 X
        $local\,minimum$    # 한 번
        $local\;minimum$    # 두 번
        $local\quad min$    # 네 번
        ```
        
        $local minimum$   
        $local\,minimum$  
        $local\;minimum$  
        $local\quad min$  
        
        ```markdown
        # 곱셈 기호
        $y = A \times x + B$  
        ```
        
        $y = A \times x + B$  
        
        ```markdown
        # 아래 첨자, 위 첨자
        $a_1, a^2, a_1^2$       
        $y_i=x_i^3+x_{i-1}^2+x_{i-2}$   # 두 글자 이상 적용시 {} 사용  
        ```
        
        $a_1, a^2, a_1^2$              
        $y_i=x_i^3+x_{i-1}^2+x_{i-2}$  

        ```markdown
        # 분수 표기법
        $s^2+2s+s\over s+\sqrt s+1$ # /over 기준 왼쪽은 모두 분자, 반대는 분모
        $\frac{1+s}{s(s+2)}$        # /frac 사용 시 첫번째가 분자  
        ```markdown
            
        $s^2+2s+s\over s+\sqrt s+1$                
        $\frac{1+s}{s(s+2)}$  

        ```markdown
        # 절대값 표기법(일반적으로 | 기호)
        $\vert x \vert$
        $\left\lvert \frac{s^2+1}{s^3+2s^2+3s+1} \right\rvert$  
        ```
            
        $\vert x \vert$  
        $\left\lvert \frac{s^2+1}{s^3+2s^2+3s+1} \right\rvert$  

        ```markdown
        # sin, log 기호 사용
        $\log_{10}{(x+1)}$
        $A\sin(bx+c)$  
        ```
            
        $\log_{10}{(x+1)}$  
        $A\sin(bx+c)$  
        
        ```markdown
        # 극한, 시그마
        $\lim_{s\rarr\infin}{s^2}$                      # linear하게 표기
        $\sum_{i=0}^{\infin}{(y_i-t_i)^2}$              
        $\displaystyle\lim_{s\rarr\infin}{s^2}$         # \displaystyle 작성 시 기본형
        $\displaystyle\sum_{i=0}^{\infin}{(y_i-t_i)^2}$   
        ```
        
        $\lim_{s\rarr\infin}{s^2}$  
        $\sum_{i=0}^{\infin}{(y_i-t_i)^2}$  
        $\displaystyle\lim_{s\rarr\infin}{s^2}$  
        $\displaystyle\sum_{i=0}^{\infin}{(y_i-t_i)^2}$  
        
        ```markdown
        # 벡터 표기법
        $\vec{a}$               
        $\overrightarrow{a}$    # \vec 보다 화살표가 더 큼  
        ```
        $\vec{a}$  
        $\overrightarrow{a}$  


        ```markdwon
        # 행렬 표기법
        $\begin{matrix}1&2\\3&4\\ \end{matrix}$     # matrix 심볼
        $\begin{pmatrix}1&2\\3&4\\ \end{pmatrix}$   # &  열구분
        $\begin{bmatrix}1&2\\3&4\\ \end{bmatrix}$   # // 행 구분
        $\begin{Bmatrix}1&2\\3&4\\ \end{Bmatrix}$
        $\begin{vmatrix}1&2\\3&4\\ \end{vmatrix}$
        $\begin{Vmatrix}1&2\\3&4\\ \end{Vmatrix}$ 
        ```

        $\begin{matrix}1&2\\3&4\\ \end{matrix}$  
        $\begin{pmatrix}1&2\\3&4\\ \end{pmatrix}$  
        $\begin{bmatrix}1&2\\3&4\\ \end{bmatrix}$  
        $\begin{Bmatrix}1&2\\3&4\\ \end{Bmatrix}$  
        $\begin{vmatrix}1&2\\3&4\\ \end{vmatrix}$  
        $\begin{Vmatrix}1&2\\3&4\\ \end{Vmatrix}$  
        
        ```markdown
        # 조각함수와 같은 case 표기법
        $\vert x\vert=  # cases 심볼을 통해 작성
        \begin{cases}
        -x,\;if\;x<0\\
        +x,\;if\;x\geq0
        \end{cases}$  
        ```
        
        $\vert x\vert=
        \begin{cases}
        -x,\;if\;x<0\\
        +x,\;if\;x\geq0
        \end{cases}$  
        
        ```markdown
        # 부등호 및 유사 등호  
        ```
        
        |||||
        | --- |---| --- |---|
        |\$\larr$| ←   |\$\rarr$ or \$\to$| →|
        |\$\gt$|	>	    |\$\lt$|	<|
        |\$\geq$|	≥	|\$\leq$|	≤|
        |\$\approx$|	≈	|\$\sim$|	∼|
        |\$\cong$|	≅	|\$\approxeq$|	≊|
        |\$\simeq$          |	≃	|\$\eqsim$|	≂|
        |\$\doteq$          |	≐	|	||
        |\$\coloneq$        |	:−	|\$\eqcolon$|	−:|
        |\$\fallingdotseq$  |	≒	|\$\risingdotseq$|	≓|
        |\$\pm$             |	±	|\$\mp$|	∓|
        
        ```markdwon
        # 점  
        ```
        
        |||||
        | --- |---| --- |---|
        |\$\dot{a}$|	$\dot{a}$ |\$\cdot$|	⋅abc|
        |\$\colon$|	:abc|\\$\ldotp$|	.abc|
        |\$\cdots$|	⋯abc|\$\ddots$|	⋱abc|
        |\$\vdots$|	⋮abc|\$ldots$|	…abc|
        |\$\because$|	∵abc|\$\therefore$| ∴abc|

        ```markdwon
        # 단위기호  
        ```
        
        |||||
        | --- |---| --- |---|
        |\$\degree$          |	° |\$\Omega$|	Ω|

            

<br><br>


> # Ⅲ. 결론

Markdown은 결국 꾸미기 보다는 글쓰는 것 자체에 중점을 두는 언어이다. 

Markdown 형태로 글을 쓸때 해당 기능들이 필요할 때 쓸 수는 있겠으나, 꾸미는데 치중해 너무 남용하거나 본질에서 벗어나지 않도록 할 필요가 있겠다. 

Markdown을 활용해 글을 체계적으로 적어나아가는 습관을 들여보자!


# REFERENCES
<hr/>

Markdown Guide : https://www.markdownguide.org/  
요즘 마크다운으로만 글을 쓰는 이유 : https://lynmp.com/ko/article/en811c9dc5nc  
MarkDown을 편하게 작성하기 위한 도구들 : https://wooncloud.tistory.com/72  
MarkDown 수식 작성법 : https://velog.io/@d2h10s/LaTex-Markdown-수식-작성법  
how-can-i-escape-3-backticks-code-block-in-3-backticks-code-block : https://stackoverflow.com/questions/49267811/how-can-i-escape-3-backticks-code-block-in-3-backticks-code-block  

[1]: https://en.wikipedia.org/wiki/WYSIWYG
[2]: https://www.markdownguide.org/getting-started/
[3]: https://ko.wikipedia.org/wiki/마크업_언어
[4]: https://www.webfx.com/tools/emoji-cheat-sheet/
