def main():
  current_dir = os.getcwd()
  model_name = sys.argv[1]
  file_name = sys.argv[2]
  if model_name == '-i' and file_name == '1.0':
    print(' . ')
    print(' . ')
    print(' . ')
    print(' . ')
    os.makedirs(current_dir + "/input_image")
    print('.........input_image 폴더생성 OK')
    os.makedirs(current_dir + "/temp")
    print('.........temp 폴더생성 OK')
    os.makedirs(current_dir + "/model")
    print('.........model 폴더생성 OK')
    print(' . ')
    print(' . ')
    print(' . ')
    print(' . ')
    print("[ version : 1.0 ]")
    print(' . ')
    print(' . ')
    print('**********초기화 완료 !***********')
    print(' . ')
    print('해상도를 높일 이미지파일.png를 input_image 폴더에 넣어주세요.')
    print('사용하실 모델파일.h5 를 model 폴더에 넣어주세요.')
    print(' . ')
    print(' . ')
    print(' . ')
    print('다음부터 아래와 같은 명령어를 사용하시면 됩니다.')
    print('ver1.exe 모델파일 이미지파일')
    return
  if not os.path.exists(current_dir + "/input_image"):
    print("초기화가 필요합니다.")
    print("ver1.exe -i -1.0 옵션으로 초기화를 시켜주세요.")
    return
  if not os.path.exists(current_dir + "/temp"):
    print("초기화가 필요합니다.")
    print("ver1.exe -i -1.0 옵션으로 초기화를 시켜주세요.")
    return
  if not os.path.exists(current_dir + "/model"):
    print("초기화가 필요합니다.")
    print("ver1.exe -i -1.0 옵션으로 초기화를 시켜주세요.")
    return
  if not os.path.isfile(current_dir + '/model/' + model_name + '.h5'):
    print("모델이 존재하지 않습니다.")
    return
  if not os.path.isfile(current_dir + '/input_image/' + file_name + '.png'):
    print("이미지가 존재하지 않습니다.")
    return
    
bad_model = load_model(current_dir+'/model/'+model_name+'.h5',
custom_objects={'Subpixel': Subpixel})
test_image = cv2.imread(current_dir + "/input_image/" + file_name + ".png")
np.save(current_dir + "/temp/test.npy", test_image)
image_x = np.load(current_dir+"/temp/test.npy")
image_x = (image_x / 255) # batch normalize 되면서 들어간다.
image_pred = bad_model.predict(image_x.reshape((1, 44, 44, 3)))
image_pred = np.clip(image_pred.reshape((176, 176, 3)), 0, 1)
image_x = (image_x * 255).astype(np.uint8)
image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
image_pred = (image_pred * 255).astype(np.uint8)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('input')
plt.imshow(image_x)
plt.subplot(1, 2, 2)
plt.title('prediction')
plt.imshow(image_pred)
plt.show()

main()
