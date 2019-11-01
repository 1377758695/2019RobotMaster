root = "D:/theThirdYear/RM/task1"
f = open(root+'/label.txt', 'w')
img_folder = root + '/gray/'
for i in range(1613):
    idx = "%04d"%i
    img_path = img_folder + str(idx) + '.jpg'
    print(img_path)
    f.write(img_path+'\n')
f.close()