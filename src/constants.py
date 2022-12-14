import numpy as np
DATASETS = [
    'cifar20', 'nonliving26', 'living17', 'entity13', 
    'entity30', 'lsun-scene', 'fashion1m', 'objectnet', 
    'food-101', 'fruits360', 'office31', 'officehome', 'fashion-mnist'
]
TRUESETS = ['nonliving26', 'living17', 'entity13', 'entity30', 'fruits360', 'objectnet']
DOMAINS = {
    'nonliving26': ['imagenet', 'imagenet-sketch', 'imagenetv2', 'imagenet-c1','imagenet-c2','imagenet-c3','imagenet-c4','imagenet-c5'],
    'living17': ['imagenet', 'imagenet-sketch', 'imagenetv2', 'imagenet-c1','imagenet-c2','imagenet-c3','imagenet-c4','imagenet-c5'],
    'entity13': ['imagenet', 'imagenet-sketch', 'imagenetv2', 'imagenet-c1','imagenet-c2','imagenet-c3','imagenet-c4','imagenet-c5'],
    'entity30': ['imagenet', 'imagenet-sketch', 'imagenetv2', 'imagenet-c1','imagenet-c2','imagenet-c3','imagenet-c4','imagenet-c5'],
    'office31': ['webcam', 'dslr', 'amazon'],
    'officehome': ['art', 'clipart', 'realworld', 'product']
}
MODELS = ['ClipViTL14', 'ClipViTB32', 'ClipViTB16', 'ClipRN50x4', 'ClipRN101', 'ClipRN50']
EXPERIMENTS = ['true', 'gpt', 'true_noise', 'true_lin', 'gpt_lin']


CIFAR20_COARSE = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 'household electrical devices', 'household furniture', 'insects', 'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores', 'medium mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees', 'vehicles'
]

CIFAR20_FINE = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'
]

CIFAR20_LABELS = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
            6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
            5, 18,  8,  8, 15, 13, 14, 17, 18, 10,
            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
            10, 3,  2, 12, 12, 16, 12,  1,  9, 18, 
            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
            16, 18,  2,  4,  6, 18,  5,  5,  8, 18,
            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])


FRUITS360_OG2SUP = {
    'Apple Braeburn': 'apple',
    'Apple Crimson Snow': 'apple',
    'Apple Golden 1': 'apple',
    'Apple Golden 2': 'apple',
    'Apple Golden 3': 'apple',
    'Apple Granny Smith': 'apple',
    'Apple Pink Lady': 'apple',
    'Apple Red 1': 'apple',
    'Apple Red 2': 'apple',
    'Apple Red 3': 'apple',
    'Apple Red Delicious': 'apple',
    'Apple Red Yellow 1': 'apple',
    'Apple Red Yellow 2': 'apple',
    'Apricot': 'apricot',
    'Avocado': 'avocado',
    'Avocado ripe': 'avocado',
    'Banana': 'banana',
    'Banana Lady Finger': 'banana',
    'Banana Red': 'banana',
    'Beetroot': 'beetroot',
    'Blueberry': 'blueberry',
    'Cactus fruit': 'cactus fruit',
    'Cantaloupe 1': 'melon',
    'Cantaloupe 2': 'melon',
    'Carambula': 'star fruit',
    'Cauliflower': 'cauliflower',
    'Cherry 1': 'cherry',
    'Cherry 2': 'cherry',
    'Cherry Rainier': 'cherry',
    'Cherry Wax Black': 'cherry',
    'Cherry Wax Red': 'cherry',
    'Cherry Wax Yellow': 'cherry',
    'Chestnut': 'nut',
    'Clementine': 'orange',
    'Cocos': 'cocos',
    'Corn': 'corn',
    'Corn Husk': 'corn husk',
    'Cucumber Ripe': 'cucumber',
    'Cucumber Ripe 2': 'cucumber',
    'Dates': 'date',
    'Eggplant': 'eggplant',
    'Fig': 'fig',
    'Ginger Root': 'ginger root',
    'Granadilla': 'passion fruit',
    'Grape Blue': 'grape',
    'Grape Pink': 'grape',
    'Grape White': 'grape',
    'Grape White 2': 'grape',
    'Grape White 3': 'grape',
    'Grape White 4': 'grape',
    'Grapefruit Pink': 'grapefruit',
    'Grapefruit White': 'grapefruit',
    'Guava': 'gauva',
    'Hazelnut': 'nut',
    'Huckleberry': 'huckleberry',
    'Kaki': 'persimmon',
    'Kiwi': 'kiwi',
    'Kohlrabi': 'kohlrabi',
    'Kumquats': 'kumquat',
    'Lemon': 'lemon',
    'Lemon Meyer': 'lemon',
    'Limes': 'lime',
    'Lychee': 'lychee',
    'Mandarine': 'orange',
    'Mango': 'mango',
    'Mango Red': 'mango',
    'Mangostan': 'mangostan',
    'Maracuja': 'passion fruit',
    'Melon Piel de Sapo': 'melon',
    'Mulberry': 'mulberry',
    'Nectarine': 'nectarine',
    'Nectarine Flat': 'nectarine',
    'Nut Forest': 'nut',
    'Nut Pecan': 'nut',
    'Onion Red': 'onion',
    'Onion Red Peeled': 'onion',
    'Onion White': 'onion',
    'Orange': 'orange',
    'Papaya': 'papaya',
    'Passion Fruit': 'passion fruit',
    'Peach': 'peach',
    'Peach 2': 'peach',
    'Peach Flat': 'peach',
    'Pear': 'pear',
    'Pear 2': 'pear',
    'Pear Abate': 'pear', 
    'Pear Forelle': 'pear', 
    'Pear Kaiser': 'pear', 
    'Pear Monster': 'pear', 
    'Pear Red': 'pear', 
    'Pear Stone': 'pear', 
    'Pear Williams': 'pear',
    'Pepino': 'pepino',
    'Pepper Green': 'pepper', 
    'Pepper Orange': 'pepper', 
    'Pepper Red': 'pepper', 
    'Pepper Yellow': 'pepper', 
    'Physalis': 'groundcherry', 
    'Physalis with Husk': 'groundcherry', 
    'Pineapple': 'pineapple', 
    'Pineapple Mini': 'pineapple', 
    'Pitahaya Red': 'dragon fruit', 
    'Plum': 'plum', 
    'Plum 2': 'plum', 
    'Plum 3': 'plum', 
    'Pomegranate': 'pomegranate',
    'Pomelo Sweetie': 'pomelo', 
    'Potato Red': 'potato', 
    'Potato Red Washed': 'potato', 
    'Potato Sweet': 'potato', 
    'Potato White': 'potato', 
    'Quince': 'quince', 
    'Rambutan': 'rambutan', 
    'Raspberry': 'raspberry', 
    'Redcurrant': 'redcurrant', 
    'Salak': 'snake fruit', 
    'Strawberry': 'strawberry', 
    'Strawberry Wedge': 'strawberry', 
    'Tamarillo': 'tamarillo', 
    'Tangelo': 'tangelo', 
    'Tomato 1': 'tomato', 
    'Tomato 2': 'tomato', 
    'Tomato 3': 'tomato', 
    'Tomato 4': 'tomato', 
    'Tomato Cherry Red': 'tomato', 
    'Tomato Heart': 'tomato', 
    'Tomato Maroon': 'tomato', 
    'Tomato Yellow': 'tomato', 
    'Tomato not Ripened': 'tomato', 
    'Walnut': 'nut', 
    'Watermelon': 'melon'
}

FRUITS360_OG2SUB = {
    'Apple Braeburn': 'braeburn apple',
    'Apple Crimson Snow': 'crimson snow apple',
    'Apple Golden 1': 'golden apple',
    'Apple Golden 2': 'golden apple',
    'Apple Golden 3': 'golden apple',
    'Apple Granny Smith': 'granny smith apple',
    'Apple Pink Lady': 'pink lady apple',
    'Apple Red 1': 'red apple',
    'Apple Red 2': 'red apple',
    'Apple Red 3': 'red apple',
    'Apple Red Delicious': 'red delicious apple',
    'Apple Red Yellow 1': 'red yellow apple',
    'Apple Red Yellow 2': 'red yellow apple',
    'Apricot': 'apricot',
    'Avocado': 'avocado',
    'Avocado ripe': 'avocado',
    'Banana': 'banana',
    'Banana Lady Finger': 'lady finger banana',
    'Banana Red': 'red banana',
    'Beetroot': 'beetroot',
    'Blueberry': 'blueberry',
    'Cactus fruit': 'cactus fruit',
    'Cantaloupe 1': 'melon',
    'Cantaloupe 2': 'melon',
    'Carambula': 'star fruit',
    'Cauliflower': 'cauliflower',
    'Cherry 1': 'cherry',
    'Cherry 2': 'cherry',
    'Cherry Rainier': 'rainier cherry',
    'Cherry Wax Black': 'black cherry',
    'Cherry Wax Red': 'red cherry',
    'Cherry Wax Yellow': 'yellow cherry',
    'Chestnut': 'nut',
    'Clementine': 'orange',
    'Cocos': 'cocos',
    'Corn': 'corn',
    'Corn Husk': 'corn husk',
    'Cucumber Ripe': 'cucumber',
    'Cucumber Ripe 2': 'cucumber',
    'Dates': 'date',
    'Eggplant': 'eggplant',
    'Fig': 'fig',
    'Ginger Root': 'ginger root',
    'Granadilla': 'granadilla',
    'Grape Blue': 'blue grape',
    'Grape Pink': 'pink grape',
    'Grape White': 'white grape',
    'Grape White 2': 'white grape',
    'Grape White 3': 'white grape',
    'Grape White 4': 'white grape',
    'Grapefruit Pink': 'pink grapefruit',
    'Grapefruit White': 'white grapefruit',
    'Guava': 'gauva',
    'Hazelnut': 'nut',
    'Huckleberry': 'huckleberry',
    'Kaki': 'kaki',
    'Kiwi': 'kiwi',
    'Kohlrabi': 'kohlrabi',
    'Kumquats': 'kumquat',
    'Lemon': 'lemon',
    'Lemon Meyer': 'meyer lemon',
    'Limes': 'lime',
    'Lychee': 'lychee',
    'Mandarine': 'orange',
    'Mango': 'mango',
    'Mango Red': 'red mango',
    'Mangostan': 'mangostan',
    'Maracuja': 'maracuja',
    'Melon Piel de Sapo': 'melon',
    'Mulberry': 'mulberry',
    'Nectarine': 'nectarine',
    'Nectarine Flat': 'flat nectarine',
    'Nut Forest': 'forest nut',
    'Nut Pecan': 'pecan nut',
    'Onion Red': 'red onion',
    'Onion Red Peeled': 'red onion',
    'Onion White': 'white onion',
    'Orange': 'orange',
    'Papaya': 'papaya',
    'Passion Fruit': 'passion fruit',
    'Peach': 'peach',
    'Peach 2': 'peach',
    'Peach Flat': 'flat peach',
    'Pear': 'pear',
    'Pear 2': 'pear',
    'Pear Abate': 'abate pear', 
    'Pear Forelle': 'forelle pear', 
    'Pear Kaiser': 'kaiser pear', 
    'Pear Monster': 'monster pear', 
    'Pear Red': 'red pear', 
    'Pear Stone': 'stone pear', 
    'Pear Williams': 'williams pear',
    'Pepino': 'pepino',
    'Pepper Green': 'green pepper', 
    'Pepper Orange': 'orange pepper', 
    'Pepper Red': 'red pepper', 
    'Pepper Yellow': 'yellow pepper', 
    'Physalis': 'groundcherry', 
    'Physalis with Husk': 'groundcherry', 
    'Pineapple': 'pineapple', 
    'Pineapple Mini': 'mini pineapple', 
    'Pitahaya Red': 'dragon fruit', 
    'Plum': 'plum', 
    'Plum 2': 'plum', 
    'Plum 3': 'plum', 
    'Pomegranate': 'pomegranate',
    'Pomelo Sweetie': 'pomelo', 
    'Potato Red': 'red potato', 
    'Potato Red Washed': 'red potato', 
    'Potato Sweet': 'sweet potato', 
    'Potato White': 'white potato', 
    'Quince': 'quince', 
    'Rambutan': 'rambutan', 
    'Raspberry': 'raspberry', 
    'Redcurrant': 'redcurrant', 
    'Salak': 'salak', 
    'Strawberry': 'strawberry', 
    'Strawberry Wedge': 'strawberry', 
    'Tamarillo': 'tamarillo', 
    'Tangelo': 'tangelo', 
    'Tomato 1': 'tomato', 
    'Tomato 2': 'tomato', 
    'Tomato 3': 'tomato', 
    'Tomato 4': 'tomato', 
    'Tomato Cherry Red': 'cherry tomato', 
    'Tomato Heart': 'heart tomato', 
    'Tomato Maroon': 'maroon tomato', 
    'Tomato Yellow': 'yellow tomato', 
    'Tomato not Ripened': 'unripe tomato', 
    'Walnut': 'nut', 
    'Watermelon': 'melon'
}

OBJECTNET_MAP = {
   "garment": [
      "Dress",
      "Jeans",
      "Skirt",
      "Suit jacket",
      "Sweater",
      "Swimming trunks",
      "T-shirt"
   ],
   "soft furnishings, accessories": [
      "Bath towel",
      "Desk lamp",
      "Dishrag or hand towel",
      "Doormat",
      "Lampshade",
      "Paper towel",
      "Pillow"
   ],
   "accessory, accoutrement, accouterment": [
      "Backpack",
      "Dress shoe (men)",
      "Helmet",
      "Necklace",
      "Plastic bag",
      "Running shoe",
      "Sandal",
      "Sock",
      "Sunglasses",
      "Tie",
      "Umbrella",
      "Winter glove"
   ],
   "appliance": [
      "Coffee/French press",
      "Fan",
      "Hair dryer",
      "Iron (for clothes)",
      "Microwave",
      "Portable heater",
      "Toaster",
      "Vacuum cleaner"
   ],
   "equipment": [
      "Cellphone",
      "Computer mouse",
      "Keyboard",
      "Laptop (open)",
      "Monitor",
      "Printer",
      "Remote control",
      "Speaker",
      "Still Camera",
      "TV",
      "Tennis racket",
      "Weight (exercise)"
   ],
   "furniture, piece of furniture, article of furniture": [
      "Bench",
      "Chair"
   ],
   "toiletry, toilet articles": [
      "Band Aid",
      "Lipstick"
   ],
   "wheeled vehicle": [
      "Basket",
      "Bicycle"
   ],
   "cooked food, prepared food": [
      "Bread loaf"
   ],
   "produce, green goods, green groceries, garden truck": [
      "Banana",
      "Lemon",
      "Orange"
   ],
   "beverage, drink, drinkable, potable": [
      "Drinking Cup"
   ]
}