import math
import queue
import random
import time
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthpy.plot as ep
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from xarray.core.utils import OrderedSet

mpl.use('TkAgg')

shorelineUpperThreshold = 0.3
shorelineLowerThreshold = -0.3
thresholdStep = 2
clipSize = 4  # число, определяющее, какая часть изображения будет отброшена при сжатии
thickenPoints = False


def open(path):
    if len(sys.argv) > 1:
        if "-ds" in sys.argv[2:]:  # означает downscale, обрезаем снимок до меньших размеров в центре
            # (убираем по 1/clipSize с каждой стороны по горизонтали и вертикали)
            band = rxr.open_rasterio(path, masked=True).squeeze()
            height = band.shape[0]
            width = band.shape[1]
            # print(band)
            # print(band.shape)
            # print("Bounding coordinates for image:", band.x[int(width / clipSize)].values, band.y[int(height / clipSize)].values,
            #       band.x[int(width / clipSize * (clipSize - 1))].values, band.y[int(height / clipSize * (clipSize - 1))].values)
            croppedBand = band.rio.clip_box(minx=band.x[int(width / clipSize)].values,
                                            miny=band.y[int(height / clipSize * (clipSize - 1))].values,
                                            maxx=band.x[int(width / clipSize * (clipSize - 1))].values,
                                            maxy=band.y[int(height / clipSize)].values)
            # print(croppedBand)
            return croppedBand
    return rxr.open_rasterio(path, masked=True).squeeze()


def getNeighbors(matrix, i, j, len=1):
    # ns = []
    # for dx in range(0 - len, 1 + len):
    #     for dy in range(0 - len, 1 + len):
    #         rangeX = range(0, matrix.shape[0])
    #         rangeY = range(0, matrix.shape[1])
    #         (newX, newY) = (i + dx, j + dy)
    #         if (newX in rangeX) and (newY in rangeY) and (dx, dy) != (0, 0):
    #             ns.append(matrix[newX][newY])
    ns = matrix[max(i - len, 0):min(i + len + 1, matrix.shape[0]),
         max(j - len, 0):min(j + len + 1, matrix.shape[1])].flatten()
    return ns


def getThreshold(matrix, minV, maxV, step):
    tv = []
    for i in range(0, matrix.shape[0], step):
        for j in range(0, matrix.shape[1], step):
            if minV <= matrix[i][j] <= maxV:
                tv.append((i, j))
    return tv


def getThresholdedNeighbors(matrix, i, j, len=1):
    ns = []
    for dx in range(0 - len, 1 + len):
        for dy in range(0 - len, 1 + len):
            rangeX = range(0, matrix.shape[0])
            rangeY = range(0, matrix.shape[1])
            (newX, newY) = (i + dx, j + dy)
            if (newX in rangeX) and (newY in rangeY) and (dx, dy) != (0, 0) and (
                    shorelineLowerThreshold <= matrix[newX][newY] <= shorelineUpperThreshold):
                ns.append((newX, newY))
    return ns


class OrderedSetQueue(queue.Queue):
    def __init(self, maxsize):
        self.queue = OrderedSet()

    def __put(self, item):
        self.queue.add(item)

    def __get(self):
        return self.queue.pop()


def markShoreline(matrix, i, j, first=False):
    if not first and shorelines[i][j] == 1:
        # не маркируем соседей если достигли ещё одну точку береговой линии, полученную при первоначальном
        # обходе сеткой, иначе ошибка с переполнением рекурсии а та точка должна сама пометить всех соседей до этого,
        # так что можно не продолжать
        return
    else:
        if thresholdStep > 2:
            neighbors = getNeighbors(matrix, i, j, 2)
            if any(val > shorelineUpperThreshold for val in neighbors) and any(
                    val < shorelineLowerThreshold for val in neighbors):
                shorelines[i][j] = 1
            neighbors = getThresholdedNeighbors(MNDWI, x, y, 4)
            for neighbor in neighbors:
                if shorelines[neighbor[0]][neighbor[1]] == 0:
                    queue.put(neighbor)
            # if all((shorelines[n[0]][n[1]] == 1) for n in neighbors):
            #     return
            # else:
            #     for neighbor in neighbors:
            #         markShoreline(MNDWI, neighbor[0], neighbor[1])
        else:  # thresholdStep = 2
            neighbors = getNeighbors(matrix, i, j, 2)
            if any(val > shorelineUpperThreshold for val in neighbors) and any(
                    val < shorelineLowerThreshold for val in neighbors):
                shorelines[i][j] = 1
                thrNeighbors = getThresholdedNeighbors(matrix, i, j, 2)
                for t in thrNeighbors:
                    if matrix[t[0]][t[1]] != 1:
                        ns = getNeighbors(matrix, t[0], t[1], 2)
                        if any(val > shorelineUpperThreshold for val in ns) and any(
                                val < shorelineLowerThreshold for val in ns):
                            shorelines[t[0]][t[1]] = 1


def thickenPointsFun(matrix, size=4):
    if thickenPoints:
        tuple = np.where(matrix == 1)
        xs = tuple[0]
        ys = tuple[1]
        # увеличение точек чтобы их было видно при отображении
        for k in range(xs.size):
            x = xs[k]
            y = ys[k]
            for i in range(size):
                for j in range(size):
                    for l in range(size):
                        try:
                            matrix[x - i][y - i + l] = 1
                            try:
                                matrix[x + i][y - i + l] = 1
                                try:
                                    matrix[x - i][y + i - l] = 1
                                    try:
                                        matrix[x + i][y + i - l] = 1
                                        try:
                                            matrix[x - i + l][y - i] = 1
                                            try:
                                                matrix[x + i - l][y - i] = 1
                                                try:
                                                    matrix[x - i + l][y + i] = 1
                                                    try:
                                                        matrix[x + i - l][y + i] = 1
                                                    except IndexError:
                                                        pass
                                                except IndexError:
                                                    pass
                                            except IndexError:
                                                pass
                                        except IndexError:
                                            pass
                                    except IndexError:
                                        pass
                                except IndexError:
                                    pass
                            except IndexError:
                                pass
                        except IndexError:
                            pass


def compress(image, srcSize, destSize, steps):
    transformations = []
    segmentation = getSegmentation(image, srcSize, destSize, steps)
    print("Starting compression of image after segmentation")
    for i in range(image.shape[0] // destSize):
        transformations.append([])
        for j in range(image.shape[1] // destSize):
            transformations[i].append(None)
            minDist = float('inf')
            block = image[i * destSize: (i + 1) * destSize, j * destSize: (j + 1) * destSize]
            for k, l, dir, angle, tr in segmentation:
                dist = np.sum(np.square(block - tr))
                if dist < minDist:
                    minDist = dist
                    transformations[i][j] = (k, l, dir, angle)
    return transformations


def getSegmentation(image, srcSize, destSize, steps):
    print("Starting segmentation of image")
    dirs = [-1, 1]
    angles = [0, 90, 180, 270]
    transforms = [[dir, angle] for dir in dirs for angle in angles]
    sizeFactor = srcSize // destSize
    segments = []
    for i in range((image.shape[0] - srcSize) // steps + 1):
        for j in range((image.shape[1] - srcSize) // steps + 1):
            seg = shrink(image[i * steps: i * steps + srcSize, j * steps: j * steps + srcSize], sizeFactor)
            # for dir, angle in transforms:
            dir, angle = random.choice(transforms)
            segments.append((i, j, dir, angle, transform(seg, dir, angle)))
    return segments


def decompress(transformations, srcSize, destSize, steps, iters=8):
    sizeFactor = srcSize // destSize
    height = len(transformations) * destSize
    width = len(transformations[0]) * destSize
    iterations = [np.random.randint(0, 256, (height, width))]
    image = np.zeros((height, width))
    for iter in range(iters):
        print("Doing decompression iteration", iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                k, l, dir, angle = transformations[i][j]
                S = shrink(iterations[-1][k * steps: k * steps + srcSize, l * steps: l * steps + srcSize], sizeFactor)
                D = transform(S, dir, angle)
                image[i * destSize: (i + 1) * destSize, j * destSize: (j + 1) * destSize] = D
        iterations.append(image)
        image = np.zeros((height, width))
    return iterations


def shrink(image, factor):
    res = np.zeros((image.shape[0] // factor, image.shape[1] // factor))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = np.mean(image[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])
    return res


def transform(image, dir, angle):
    return rotate(flip(image, dir), angle)


def flip(image, dir):
    return image[::dir, :]


def rotate(image, angle):
    res = np.copy(image)
    for _ in range(angle, 0, -90):
        res = np.rot90(res)
    return res


def plotIterations(iters, target=None):
    rows = math.ceil(np.sqrt(len(iters)))
    cols = rows
    fig, axs = plt.subplots(cols + 1, rows, figsize=(12, 8))
    res = np.zeros_like(shorelines)
    for i, image in enumerate(iters):
        if i > 0:
            lowestValue = np.min(image)
            if np.count_nonzero(image == lowestValue) > np.count_nonzero(image > lowestValue):
                ep.plot_bands(image, cmap="binary", ax=axs[i // rows][i % cols])
                if i == (len(iters) - 1):   # делаем бинарное изображение на основе последней итерации
                    res = np.where(image > lowestValue, 1, 0)
            else:
                ep.plot_bands(image, cmap="Greys_r", ax=axs[i // rows][i % cols])
                if i == (len(iters) - 1):   # делаем бинарное изображение на основе последней итерации
                    highestValue = np.max(image)
                    res = np.where(image < highestValue, 1, 0)
        else:
            ep.plot_bands(image, cmap="binary", ax=axs[i // rows][i % cols])
        plt.title(str(i))
    ep.plot_bands(shorelines, cmap="binary", ax=axs[rows][0], title="Original")
    ep.plot_bands(target, cmap="binary", ax=axs[rows][1], title="Shrinked")
    ep.plot_bands(res, cmap="binary", ax=axs[rows][2], title="Result")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folderName = sys.argv[1]
    else:
        folderName = "test1"
    fileList = ["../tests/" + folderName + "/B1.TIF", "../tests/" + folderName + "/B2.TIF",
                "../tests/" + folderName + "/B3.TIF", "../tests/" + folderName + "/B5.TIF"]

    if len(sys.argv) > 1 and "-ds" in sys.argv[2:]:
        if sys.argv.index("-ds") < (len(sys.argv) - 1):
            try:
                clip = float(sys.argv[sys.argv.index("-ds") + 1])
                print("Got custom clip value from console")
                if 2 < clip <= 6:
                    clipSize = clip
                    print("Clipsize:", clipSize)
                else:
                    print("Incorrect clip value: must be integer from 3 to 6 incl.\nsetting default clipsize of 4")
            except ValueError:
                print("Got not an int/float value after -ds key, ignored")

    allBands = []
    for i, band in enumerate(fileList):
        allBands.append(open(band))  # * 0.0000275) + -0.2)
        allBands[i]["band"] = i + 1
    allBands[3]["band"] = 5
    allBands = xr.concat(allBands, dim="band")

    # print(allBands)

    allBands.plot.imshow(col="band", col_wrap=2, cmap="Greys_r")
    plt.show()

    if len(sys.argv) > 1 and "-ds" in sys.argv[2:]:
        # print(allBands.values)
        ep.plot_rgb(allBands.values, rgb=(2, 1, 0), title="RGB image",
                    stretch=True)  # killed по памяти на 7400х8200 поэтому только на уменьшенных размерах
        plt.show()

    if len(sys.argv) > 1:
        if "-qa" in sys.argv[2:]:
            path = "../tests/" + folderName + "/QA.TIF"
            qa = open(path).astype(int)

            cloud = np.where(np.bitwise_and(qa, 0b1000) != 0, True, False)
            mediumCloudConfidence = np.where(np.bitwise_and(qa, 0b1000000000) != 0, True, False)
            # highCloudConfidence = np.where(np.bitwise_and(qa, 0b1100000000) != 0, True, False) - уже входит в определение cloud
            # lowCloudConfidence = np.where(np.bitwise_and(qa, 0b0100000000) != 0, True, Fals
            cloudShadow = np.where(np.bitwise_and(qa, 0b10000) != 0, True, False)
            mediumCloudShadowConfidence = np.where(np.bitwise_and(qa, 0b100000000000) != 0, True, False)
            # highCloudShadowConfidence = np.where(np.bitwise_and(qa, 0b110000000000) != 0, True, False) - уже входит в определение cloudShadow
            # lowCloudShadowConfidence = np.where(np.bitwise_and(qa, 0b010000000000) != 0, True, False)

            # print(qa.sel(x=5000, y=1500, method='nearest').values)
            # print(np.unique(qa.values))
            # print(np.unique(cloud))
            # print(np.unique(cloudShadow))
            # print(np.unique(mediumCloudConfidence))
            # print(np.unique(mediumCloudShadowConfidence))

            # print(np.unique(highCloudConfidence))
            # print(np.unique(mediumCloudConfidence))
            # print(np.unique(lowCloudConfidence))
            # print(np.unique(highCloudShadowConfidence))
            # print(np.unique(lowCloudShadowConfidence))

            # print(cloud)
            # print(cloudShadow)
            # print(mediumCloudConfidence)
            # print(mediumCloudShadowConfidence)

            allClouds = np.any((cloud, cloudShadow, mediumCloudConfidence, mediumCloudShadowConfidence), axis=0)
            del cloud
            del cloudShadow
            del mediumCloudConfidence
            del mediumCloudShadowConfidence

            # mask = np.logical_not(allClouds)
            # del allClouds
            # print(np.unique(allClouds))
            fig, axs = plt.subplots(1, 2, figsize=(12, 8))
            qaBand = axs[0].imshow(qa, cmap="cool")
            axs[0].set_title("QA Band")
            # plt.imshow(qa, cmap="cool")
            # plt.show()
            del qa
            im = axs[1].imshow(allClouds, cmap="Greys_r")
            axs[1].set_title("Cloud mask")
            # plt.imshow(mask, cmap="Greys_r")
            plt.show()

            allBands = allBands.where(~allClouds)
            # del mask
            del allClouds
            ep.plot_bands(allBands)
            plt.show()

            if "-ds" in sys.argv[2:]:
                ep.plot_rgb(allBands.values, rgb=(2, 1, 0), title="RGB image",
                            stretch=True)
                plt.show()
        else:
            print("-qa key not entered - not checking for qa file")
    else:
        print("-qa key not entered - not checking for qa file")

    print("Calculating MNDWI")
    # MNDWI = (GREEN - SWIR) / (GREEN + SWIR)
    MNDWI = ((allBands[1] - allBands[3]) / (allBands[1] + allBands[3])).values
    print("MNDWI values:", MNDWI.min(), 'to', MNDWI.max(), 'mean:', MNDWI.mean())
    print("MNDWI shape:", MNDWI.shape)
    ep.plot_bands(MNDWI, cmap="RdBu", vmin=-1, vmax=1, title="MNDWI")
    plt.show()

    '''
    Производительность на примере вызова python3 main.py test5 -qa -ds 4:
        MNDWI.shape: (3731, 4107) - 15323217 пикселей
        thresholdStep = 1: 217.83 секунд, всего отмечено 51179 пикселей
        thresholdStep = 2: 55.43 + 51.68 = 107.11 секунд, отмечено 50603, не совпало 576 пикселей с thresholdStep = 1
        thresholdStep = 16: работает медленно и неточно
        
    На примере вызова python3 main.py test1 -qa:
        MNDWI.shape: (7061, 8011) - 56565671 пикселей
        thresholdStep = 1: 769.63 секунд, отмечено 75226 пикселей
        thresholdStep = 2: 193.38 + 107.38 = 300.76 секунд, отмечено 72670 пикселей,
                           не совпало 2556 пикселей с thresholdStep = 1
    '''

    print("Extracting shorelines")
    start1 = time.time()
    shorelines = np.zeros((MNDWI.shape[0], MNDWI.shape[1]))
    thresholdedValueCoords = getThreshold(MNDWI, shorelineLowerThreshold, shorelineUpperThreshold, thresholdStep)
    startingIndexes = []
    for index, tuple in enumerate(thresholdedValueCoords):
        x = tuple[0]
        y = tuple[1]
        neighbors = getNeighbors(MNDWI, x, y, 2)
        if any(val > shorelineUpperThreshold for val in neighbors) and any(
                val < shorelineLowerThreshold for val in neighbors):
            shorelines[x][y] = 1
            startingIndexes.append((x, y))
    print("Got", len(startingIndexes), "base shoreline pixels from extracting with thresholdStep =", thresholdStep)
    if thresholdStep > 1:
        end1 = time.time()
        print("Time spent extracting shorelines with step", str(thresholdStep) + ':', end1 - start1)
        print("Starting traversing through the shoreline")
        start2 = time.time()
        if thresholdStep > 2:
            queue = OrderedSetQueue()
            for coords in startingIndexes:
                queue.put((coords[0], coords[1]))
            for index, tuple in enumerate(startingIndexes):
                x = tuple[0]
                y = tuple[1]
                neighbors = getThresholdedNeighbors(MNDWI, x, y, 1)
                for neighbor in neighbors:
                    queue.put((neighbor[0], neighbor[1]))
                # снимаем пару координат из thresholdedValueCoords
                n = queue.get()
                markShoreline(MNDWI, n[0], n[1], first=True)
            iternum = 1
            print("Start emptying queue")
            while not queue.qsize() == 0:
                n = queue.get()
                iternum += 1
                if iternum % 1000 == 0:
                    print("Iteration №" + str(iternum), np.count_nonzero(shorelines))
                markShoreline(MNDWI, n[0], n[1], first=False)
                if queue.qsize == 0:
                    print("Last iteration:", iternum)
        else:
            for coords in startingIndexes:
                neighbors = getThresholdedNeighbors(MNDWI, coords[0], coords[1], 1)
                for n in neighbors:
                    markShoreline(MNDWI, n[0], n[1], True)
    else:
        start2 = start1
    end2 = time.time()
    print("Time spent extracting shorelines:", end2 - start2)
    print("Count of shoreline pixels:", np.count_nonzero(shorelines))

    if len(sys.argv) > 1:
        if "-cmp" in sys.argv[2:] and thresholdStep > 1:
            # для сравнения сделаем полный обход изображения с thresholdStep = 1 и сравним значения
            print("Making full shoreline extraction for comparison")
            start = time.time()
            extraShorelines = np.zeros((MNDWI.shape[0], MNDWI.shape[1]))
            extraThresholdedValueCoords = getThreshold(MNDWI, shorelineLowerThreshold, shorelineUpperThreshold, 1)
            for index, tuple in enumerate(extraThresholdedValueCoords):
                x = tuple[0]
                y = tuple[1]
                neighbors = getNeighbors(MNDWI, x, y, 2)
                if any(val > shorelineUpperThreshold for val in neighbors) and any(
                        val < shorelineLowerThreshold for val in neighbors):
                    extraShorelines[x][y] = 1
            end = time.time()
            print("Time spent on extracting shoreline with step = 1:", end - start)
            print("Shoreline pixels in shoreline with step = 1:", np.count_nonzero(extraShorelines))
            comparison = np.where(extraShorelines != shorelines, 1, 0)
            print(np.unique(comparison))
            nonMatch = []
            for i in range(0, comparison.shape[0], 1):
                for j in range(0, comparison.shape[1], 1):
                    if comparison[i][j] == 1:
                        nonMatch.append((i, j))
            if len(nonMatch) < 500:
                print(nonMatch)
            print("Non-matching ", len(nonMatch), "pixels")
            thickenPointsFun(comparison, 4)

            fig, axs = plt.subplots(1, 3)
            ep.plot_bands(shorelines, cmap="binary", ax=axs[0],
                          title=("Shorelines with thresholdStep =" + str(thresholdStep)))
            ep.plot_bands(extraShorelines, cmap="binary", ax=axs[1], title="Shorelines with thresholdStep = 1")
            ep.plot_bands(comparison, cmap="binary", ax=axs[2], title="Difference between shorelines")
            plt.show()

    print("Preparing output of shorelines")
    outputShorelines = np.copy(shorelines)

    if thickenPoints:
        thickenPointsFun(outputShorelines, 4)
        # добавляем сетку шагом thresholdStep (если больше 16)
        if thresholdStep > 16:
            print(outputShorelines.shape[0], outputShorelines.shape[1])
            for i in range(0, outputShorelines.shape[0], thresholdStep):
                x = np.ones(outputShorelines.shape[1])
                outputShorelines[i, :] = x
                # for j in range(outputShorelines.shape[1]):
                #     outputShorelines[i][j] = 1
            for i in range(0, outputShorelines.shape[1], thresholdStep):
                x = np.ones(outputShorelines.shape[0])
                outputShorelines[:, i] = x
                # for j in range(outputShorelines.shape[0]):
                #     outputShorelines[j][i] = 1

    fig, axs = plt.subplots(1, 2)
    ep.plot_bands(outputShorelines, cmap="binary", ax=axs[0], title="Shorelines")
    ep.plot_bands(MNDWI, cmap="RdBu", vmin=-1, vmax=1, ax=axs[1], title="MNDWI")

    plt.show()

    print("Shrinking shoreline image")
    image = shrink(np.where(shorelines == 1, 255, 0), 4)
    ep.plot_bands(image, cmap="binary", title="Lower-resolution image")

    plt.show()

    # image = np.copy(shorelines)

    '''
    Производительность на примере вызова python3 main.py test1 -qa -ds 3:
        MNDWI shape: (2355, 2671)
        Выделение береговой линии; 21.39 + 8.30 = 29.69 секунд, получено 6806 пикселей береговой линии
        Время, потраченное на сжатие изеображения; 729.80 секунд, на восстановление - 22.29 секунд.
    '''

    print("Starting fractal compression of image")
    start = time.time()
    compressedTrs = compress(image, 8, 4, 8)
    print("Time spent compressing image:", time.time() - start)
    print("Starting fractal decompression of image")
    start = time.time()
    decompressedIters = decompress(compressedTrs, 8, 4, 8)
    print("Time spent decompressing image:", time.time() - start)
    plotIterations(decompressedIters, image)

    plt.show()
