import base64

import requests


if __name__ == '__main__':
    url = 'http://0.0.0.0:8080/api/control/scribble'

    control_imgpath = './src/test_imgs/user_3.png'
    with open(control_imgpath, "rb") as f:
        control_b64 = base64.b64encode(f.read()).decode('UTF-8')

    data = {'prompts': 'hot air balloon, best quality, extremely detailed, sunset, beach',
            'neg_prompts': 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
            'image': control_b64,
            'steps': 20,
            'scale': 9.0}

    for i in range(10):
        res = requests.post(url, json=data)
    print(res.json()['image'])