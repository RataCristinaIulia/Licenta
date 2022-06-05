import random
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

train_x = []
train_y = []
test_x = []
test_y = []
k = 3

population = []
new_population = []
no_individuals = 100
mutation_chance = 0.01
cross_chance = 0.3

eval = []
current_index = 0


def read_keel_file(file_path):
    with open(file_path, 'r') as file:
        # citim linii pana gasim o linie care incepe cu "@data"
        while True:
            line = file.readline()
            if line.startswith("@data"):
                break

        instances = []
        classes = []
        while True:
            line = file.readline()

            # conditie pt EOF
            if len(line) == 0:
                break
            # elimin \n de la final
            if line[-1] == '\n':
                line = line[:-1]

            values = line.split(',')
            values = list(map(lambda x: __processField(x), values))

            instances.append(values[:-1])
            classes.append(values[-1])

        return instances, classes


def __processField(field_value):
    # transform stringuri in ints sau floats
    try:
        value = float(field_value)
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        return field_value


# imparte instantele in set de antrenare si set de testare
# percentage = procentul ca instantele sa faca parte din setul de antrenare
def split_instances(instances, classes, percentage):
    nb_train_set = int(len(instances) * percentage)
    # nb_train_set = len(instances) - nb_test_set

    indexes = list(range(len(instances)))

    # amestex indecsii pt a randomiza listele
    random.shuffle(indexes)

    shuffled_instances = [instances[i] for i in indexes]
    shuffled_classes = [classes[i] for i in indexes]

    # return instante_antrenare, instante_testare, clase_antrenare, clase_testare

    return shuffled_instances[:nb_train_set], shuffled_instances[nb_train_set:], shuffled_classes[
                                                                                 :nb_train_set], shuffled_classes[
                                                                                                 nb_train_set:]


def generate_procentages(train_set):
    procentages = []
    for i in range(0, len(train_set)):
        procentage = random.uniform(0, 1)
        procentages.append(procentage)
    return procentages


def generate_population(train_set):
    for i in range(0, no_individuals):
        eval.append(1)
        procentages = generate_procentages(train_set)
        population.append(procentages)


def my_weights(weights):
    global train_x, test_x, k, current_index
    # print("Weights: \n", weights)

    indexes_nearest_neighbors = get_indexes_nearest_neighbors(train_x, test_x, k)
    procentages = population[current_index]
    current_index += 1

    for i in range(0, len(weights)):
        for j in range(0, len(weights[i])):
            weights[i][j] = weights[i][j] * procentages[indexes_nearest_neighbors[i][j]]

    return weights


def get_indexes_nearest_neighbors(train_x, test_x, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_x)

    NearestNeighbors(n_neighbors=k)
    # print("Indicii celor mai apropiati k vecini din setul de antrenare pentru toate instantele din setul de
    # testare: ")
    indexes_nearest_neighbors = neigh.kneighbors(test_x)[1]

    return indexes_nearest_neighbors


def mutation(individ):
    prob = random.uniform(0, 1)
    if prob < mutation_chance:
        rand_pos = random.randint(0, len(individ))
        new_procentage = random.uniform(0, 1)
        individ[rand_pos] = new_procentage
    return individ


def crossing(crom1, crom2):
    prob = random.uniform(0, 1)
    if prob < cross_chance:
        crossing_poz = random.randint(0, len(crom1))
        child1 = crom1[0:crossing_poz] + crom2[crossing_poz:len(crom2)]
        child2 = crom2[0:crossing_poz] + crom1[crossing_poz:len(crom1)]


def genetic_algorithm():
    global train_x, train_y, test_x, test_y, k, current_index
    # neight = KNeighborsClassifier(n_neighbors=k, weights=my_weights)
    # neight.fit(train_x, train_y)
    #
    # print("\nCalcul durata de timp...")
    # start = time.process_time()
    # score = neight.score(test_x, test_y)
    # end = time.process_time()
    # # print(f'{start}  {end}')
    # print(f'{end - start} secunde')
    #
    # print(f'Scorul este: {score}')

    generate_population(train_x)
    current_index = 0
    for i in range(0, no_individuals):
        neight = KNeighborsClassifier(n_neighbors=k, weights=my_weights)
        neight.fit(train_x, train_y)

        start = time.process_time()
        score = neight.score(test_x, test_y)
        eval[i] = score
        end = time.process_time()
        print(f'{end - start} secunde')

    t = 0
    while t != 1:
        current_index = 0
        t += 1
        # de adaugat elitismul

        # roata norocului
        total_fitness = 0
        for i in range(0, no_individuals):
            total_fitness += eval[i]

        individual_sel_prob = []
        for i in range(0, no_individuals):
            individual_sel_prob.append(eval[i] / total_fitness)
        print("individual_sel_prob", individual_sel_prob)

        accumulated_sel_prob = []
        for i in range(0, no_individuals):
            accumulated_sel_prob.append(0)
        for i in range(0, no_individuals - 1):
            accumulated_sel_prob[i + 1] = accumulated_sel_prob[i] + individual_sel_prob[i]
        print("accumulated_sel_prob", accumulated_sel_prob)

        no_selected_individuals = 0
        for i in range(0, no_individuals):
            uniform_random = random.uniform(0, 1)
            for j in range(0, no_individuals - 1):
                if accumulated_sel_prob[j] < uniform_random <= accumulated_sel_prob[j + 1]:
                    new_population.append(population[j])
                    no_selected_individuals += 1
                    break

        print("nr indivizi selectati cu roata norocului", no_selected_individuals)

        # mutatie
        # for i in range(0, no_individuals):
        #     new_population[i] = mutation(new_population[i])


if __name__ == '__main__':
    print("Citesc din fisier...")
    instances, classes = read_keel_file('shuttle.dat')
    print("Am citit datele din fisier")

    train_x, test_x, train_y, test_y = split_instances(instances, classes, 0.8)

    print(f'Lungime set de antrenare: {len(train_x)}\nLungime set de testare: {len(test_x)}')

    genetic_algorithm()
