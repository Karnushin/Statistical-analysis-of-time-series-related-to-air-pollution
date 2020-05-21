Анализ элементов примерно такой:

1. Построение точечного графика по всей выборки.
2. Количество нулевых элементов всей выборки, если есть, иначе нахождения некоторого порога весьма близкого к нулю, ниже которого элементов будет «достаточно».
3.  Гипотеза проверки однородности Колмогорова-Смирнова с окном 100 на 100, сдвигая каждый раз на единицу. Также потребуется придумать некоторую визуализацию всех этих проверок. Пока что остановимся на том, что будем принимать гипотезу, если $p-value > 5\%$, сопостовляя в соответствующем массиве $1$ данному исходу, в противном случае гипотеза отвергается и ставим $0$. Затем строим гистограмму $0-1$.
4.  Построение гистограммы всей выборки, подгонка смеси гамма распределений, используя ЕМ алгоритм + только одного распределения и критерий согласия Хи-2 и К-С
5. Построение гистограммы по сезонам и подгонка гамма распределения к ней + критерий согласия Хи-2 и К-С
2. 
1. Используя график, полученный в $1.А.$, разделяем выборку на $k$ число частей, где данный ведут себя схожим образом, так, например, для TSP визуально видно, что ее надо разбить на 3 части.
2. Провести анализ, аналогичный пункту $1$ для найденных $k$ частей.
3. 
1. Берем другой элемент и повторяем пункты $1$ и $2$.