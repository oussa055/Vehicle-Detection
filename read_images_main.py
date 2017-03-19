import os
import glob

def read_images_data():
    #cars
    directory = 'vehicles/vehicles/'
    folders = os.listdir(directory)
    cars = []
    for imtype in folders:
        cars.extend(glob.glob(directory+imtype+'/*'))
    print ('Number of cars = ', len(cars))
    with open("cars.txt", 'w') as f:
        for fn in cars:
            f.write(fn + '\n')
    #Not cars 
    directory = 'non-vehicles/non-vehicles/'
    folders = os.listdir(directory)
    notcars = []
    for imtype in folders:
        notcars.extend(glob.glob(directory+imtype+'/*'))
    print ('Number of non-cars = ', len(notcars))
    with open("notcars.txt", 'w') as f:
        for fn in notcars:
            f.write(fn + '\n')
    return cars, notcars