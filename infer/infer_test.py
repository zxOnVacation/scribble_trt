import base64
import requests


if __name__ == '__main__':
    # url = 'http://0.0.0.0:8080/api/control/scribble'
    url = 'http://controlnet-scribble.middle-layer-prod.svc.cluster.local:8080/api/control/scribble'

    # scribble图片
    control_imgpath = '/Users/zhangxiao/jike/dev/scribble_trt/src/test_imgs/user_1.png'
    control_imgpath = '/Users/zhangxiao/jike/dev/ControlNet/scribble_lyc.jpg'
    with open(control_imgpath, "rb") as f:
        control_b64 = base64.b64encode(f.read()).decode('UTF-8')

    data = {'prompts': 'a beautiful women, person, best quality, extremely detailed, sunset, beach',
            'neg_prompts': 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
            'image': control_b64,
            'steps': 20,
            'scale': 9.0
            }

    res = requests.post(url, json=data)
    # 返回图片的base64 string
    img_b64 = res.json()['image']
    print(img_b64)