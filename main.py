import random
import math
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class WorkWithData:
    # конструктор
    def __init__(self, classes_count, elements_count, class_colormap):
        self.classes_count = classes_count
        self.elements_count = elements_count
        self.class_colormap = class_colormap

        self.class_names = []

        self.data = []

    # генерируем список формата [[x, y, номер_класса], ...]
    def generate_data(self):
        for class_num in range(self.classes_count):
            # Choose random center of 2-dimensional gaussian
            center_x, center_y = random.random() * 5.0, random.random() * 5.0
            # Choose numberOfClassElements random nodes with RMS=0.5
            for rowNum in range(self.elements_count):
                self.data.append([random.gauss(center_x, 0.5), random.gauss(center_y, 0.5), class_num])

    # изображаем классы и их элементы на графике
    def add_data_to_plot(self):
        plt.scatter([self.data[i][0] for i in range(len(self.data))],
                    [self.data[i][1] for i in range(len(self.data))],
                    c=[self.data[i][2] for i in range(len(self.data))],
                    cmap=self.class_colormap)

    # подсчёт среднего внутриклассового расстояния для выбранного класса
    def inclass_range(self, class_num):
        part_of_data = []
        average_range = []
        for item in self.data:
            if item[2] == class_num:
                part_of_data.append([item[0], item[1]])
        for first_dot in part_of_data:
            for second_dot in part_of_data:
                if first_dot[0] != second_dot[0] and first_dot[1] != second_dot[1]:
                    average_range.append(math.hypot(second_dot[0] - first_dot[0], second_dot[1] - first_dot[1]))
        return np.mean(average_range)

    # подсчёт среднего внешнеклассового расстояния для выбранных двух классов
    def outclass_range(self, first_class_num, second_class_num):
        first_part_of_data = []
        second_part_of_data = []
        average_range = []
        for item in self.data:
            if item[2] == first_class_num:
                first_part_of_data.append([item[0], item[1]])
            if item[2] == second_class_num:
                second_part_of_data.append([item[0], item[1]])
        for first_dot in first_part_of_data:
            for second_dot in second_part_of_data:
                average_range.append(math.hypot(second_dot[0] - first_dot[0], second_dot[1] - first_dot[1]))
        return np.mean(average_range)

    # возвращает информативность признакового пространства
    def informativeness(self):
        inclass_avg_range = []
        outclass_avg_range = []
        for i in range(self.classes_count):
            inclass_avg_range.append(self.inclass_range(i))

        for i in range(self.classes_count):
            for j in range(self.classes_count):
                if i != j:
                    outclass_avg_range.append(self.outclass_range(i, j))

        inclass_avg_range = np.mean(inclass_avg_range)
        outclass_avg_range = np.mean(outclass_avg_range)

        return outclass_avg_range / inclass_avg_range

    # метод ближайшего соседа
    def nearest_neighbour(self, required_point):
        distance = None
        required_point_class = None
        for point in self.data:
            new_distance = math.hypot(point[0] - required_point[0], point[1] - required_point[1])
            if distance is None:
                distance = new_distance
            if new_distance < distance:
                distance = new_distance
                required_point_class = point[2]
        return [distance, required_point_class]

    # метод k-ближайших соседей
    def k_nearest_neighbour(self, required_point, k):
        distances = []
        result = []
        for point in self.data:
            new_distance = math.hypot(point[0] - required_point[0], point[1] - required_point[1])
            distances.append([new_distance, point[2]])
        distances.sort()
        for point_class in range(self.classes_count):
            sum_of_weight = 0
            for elem in distances[:k]:
                if elem[1] == point_class:
                    sum_of_weight += 1
            result.append([sum_of_weight, point_class])
        return max(result)

    # метод взвешенного голосования
    def weighed_nearest_neighbour(self, required_point, k):
        distances = []
        result = []
        for point in self.data:
            new_distance = math.hypot(point[0] - required_point[0], point[1] - required_point[1])
            distances.append([new_distance, point[2]])
        distances.sort()
        weights = [[1 / (distances[i][0]) ** 2, distances[i][1]] for i in range(k)]
        for point_class in range(self.classes_count):
            sum = 0
            for elem in weights:
                if elem[1] == point_class:
                    sum += elem[0]
            result.append([sum, point_class])
        return max(result)


def divide_prints():
    print('-' * 70)


if __name__ == "__main__":

    # количество классов
    class_count = 3
    # количество элементов
    elements_count = 40

    # для классификации (доделать)
    colors = {
        0: ['#FF0000', 'красный'],
        1: ['#00FF00', 'зеленый'],
        2: ['#0000FF', 'синий']
    }

    # задаём цвета классов, количество цветов = количеству классов
    colors_of_classes = ListedColormap([colors[0][0],  # 0 - красный,
                                        colors[1][0],  # 1 - зеленый,
                                        colors[2][0],  # 2 - синий
                                        ])

    # начинаем работать с данными
    example = WorkWithData(class_count, elements_count, colors_of_classes)
    example.generate_data()
    example.add_data_to_plot()
    plt.show()

    divide_prints()
    # информативность признакового пространства
    print("Информативность признакового пространства: {0}".format(example.informativeness()))

    divide_prints()

    # объявление новой точки
    random_point = [random.random() * 5.0, random.random() * 5.0]
    print("Координаты новой точки: X: {0}, Y: {1}".format(random_point[0], random_point[1]))

    divide_prints()

    # классификация методом ближайшего соседа
    result = example.nearest_neighbour(random_point)
    print("Метод ближайшего соседа, k = 1 \n"
          "Дистанция до ближайшей точки: {0}, \n"
          "Определение принадлежности классу: {1}".format(result[0], colors[result[1]][1]))

    divide_prints()

    # классификация методом k-ближайших соседей
    # переменная k - число соседей
    k = 7
    result = example.k_nearest_neighbour(random_point, k)
    print("Метод k-ближайших соседей, k = {0} \n"
          "Среди всех соседей {1} точек класса: {2}\n"
          "Определение принадлежности классу: {2}".format(k, result[0], colors[result[1]][1]))

    divide_prints()

    # классификация методом взвешенного голосования
    # переменная k - число соседей
    result = example.weighed_nearest_neighbour(random_point, k)
    print("Метод k-взвешенных соседей, k = {0} \n"
          "Наибольший вес {1} у соседей класса: {2}\n"
          "Определение принадлежности классу: {2}".format(k, result[0], colors[result[1]][1]))

    divide_prints()

    # наглядное добавление новой точки черным цветом на график
    example.add_data_to_plot()
    plt.scatter(random_point[0], random_point[1], c='BLACK')
    plt.show()
