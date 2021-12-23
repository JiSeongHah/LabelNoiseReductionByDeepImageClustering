import torch

print(torch.cuda.is_available())

print(torch.cuda.get_device_name())

print(torch.cuda.device_count())

a = '1baeed0f32ea23eed2c1166ab6b92b086f181a030c9a4e59bf424515c558bf1c57c5f6ce077c2f94c12644eb8224f6034b5d724d726636454428c12459f2028f'
b = '1baeed0f32ea23eed2c1166ab6b92b086f181a030c9a4e59bf424515c558bf1c57c5f6ce077c2f94c12644eb8224f6034b5d724d726636454428c12459f2028f'

if a == b:
    print('yes')
else:
    print('no')
