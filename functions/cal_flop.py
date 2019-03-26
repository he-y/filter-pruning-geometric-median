# calculate flop for ResNet on imagenet


def basic(layer, layer_index, channel, width, prune_rate):
    flop = 0

    def even_odd(flop, index, add, rate):
        if index % 2 == 0:
            flop += add * (prune_rate ** 2)
        elif index % 2 != 0:
            flop += add * (prune_rate)
        return flop

    for index in range(0, layer, 1):
        if index == 0:
            flop += channel[0] * 112 * 112 * 7 * 7 * 3 * prune_rate
        elif index in [1, 2]:
            flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate ** 2)

        elif index > 2 and index <= layer_index[0]:
            add = channel[0] * width[0] * width[0] * 9 * channel[0]
            flop = even_odd(flop, index, add, prune_rate)

        elif index > layer_index[0] and index <= layer_index[1]:
            add = channel[1] * width[1] * width[1] * 9 * channel[1]
            flop = even_odd(flop, index, add, prune_rate)

        elif index > layer_index[1] and index <= layer_index[2]:
            add = channel[2] * width[2] * width[2] * 9 * channel[2]
            flop = even_odd(flop, index, add, prune_rate)

        elif index > layer_index[2] and index <= layer_index[3]:
            add = channel[3] * width[3] * width[3] * 9 * channel[3]
            flop = even_odd(flop, index, add, prune_rate)

    less1 = channel[1] * width[1] * width[1] * 9 * channel[1] * (prune_rate) - channel[1] * width[1] * width[1] * 9 * \
            channel[0] * (prune_rate)
    less2 = channel[2] * width[2] * width[2] * 9 * channel[2] * (prune_rate) - channel[2] * width[2] * width[2] * 9 * \
            channel[1] * (prune_rate)
    less2 = channel[3] * width[3] * width[3] * 9 * channel[3] * (prune_rate) - channel[3] * width[3] * width[3] * 9 * \
            channel[2] * (prune_rate)
    flop = flop - less1 - less2
    return flop


def bottle(layer, block, channel, width, prune_rate):
    def second_block(channel, width, prune_rate):
        flop = 0
        flop += channel * width * width * 1 * 1 * (channel * 4) * prune_rate
        flop += channel * width * width * 3 * 3 * channel * (prune_rate ** 2)
        flop += (channel * 4) * width * width * 1 * 1 * channel * (prune_rate ** 2)
        return flop

    def first_block(input_channel, channel, width, prune_rate):
        flop = 0
        flop += channel * width * width * 1 * 1 * input_channel * prune_rate
        flop += channel * width * width * 3 * 3 * channel * (prune_rate ** 2)
        flop += (channel * 4) * width * width * 1 * 1 * channel * (prune_rate ** 2)
        downsample = (channel * 4) * width * width * 1 * 1 * input_channel
        flop += downsample
        return flop

    flop = 0
    flop += channel[0] * 112 * 112 * 7 * 7 * 3 * prune_rate
    #        print(flop)

    for index, num in enumerate(block):
        if index == 0:
            flop += first_block(64 * prune_rate, channel[0], width[0], prune_rate)
            flop += (num - 1) * second_block(channel[0], width[0], prune_rate)
        elif index == 1:
            flop += first_block(channel[0] * 4, channel[1], width[1], prune_rate)
            flop += (num - 1) * second_block(channel[1], width[1], prune_rate)
        elif index == 2:
            flop += first_block(channel[1] * 4, channel[2], width[2], prune_rate)
            flop += (num - 1) * second_block(channel[2], width[2], prune_rate)
        elif index == 3:
            flop += first_block(channel[2] * 4, channel[3], width[3], prune_rate)
            flop += (num - 1) * second_block(channel[3], width[3], prune_rate)
    return flop


def imagenet_flop(layer=18, prune_rate=1):
    flop = 0
    if layer == 18:
        block = [2, 2, 2, 2]
        conv_in_blcok = 2
    elif layer == 34:
        block = [3, 4, 6, 3]
        conv_in_blcok = 2
    elif layer == 50:
        block = [3, 4, 6, 3]
        conv_in_blcok = 3
    elif layer == 101:
        block = [3, 4, 23, 3]
        conv_in_blcok = 3
    elif layer == 152:
        block = [3, 8, 36, 3]
        conv_in_blcok = 3
    else:
        print("wrong layer")

    channel = [64, 128, 256, 512]
    width = [56, 28, 14, 7]

    layer_interval = [conv_in_blcok * i for i in block]
    layer_index = [sum(layer_interval[:k + 1]) for k in range(0, len(layer_interval))]
    print(layer_index)

    if layer in [18, 34]:
        flop = basic(layer, layer_index, channel, width, prune_rate)
    elif layer in [50, 101, 152]:
        flop = bottle(layer, block, channel, width, prune_rate)
        print('bottle structure')
    print(flop)
    return flop


def cal(func, layer, rate):
    print(1 - int(func(layer, rate)) / int(func(layer, 1)))


def cifar_resnet_flop(layer=110, prune_rate=1):
    '''
    :param layer: the layer of Resnet for Cifar, including 110, 56, 32, 20
    :param prune_rate: 1 means baseline
    :return: flop of the network
    '''
    flop = 0
    channel = [16, 32, 64]
    width = [32, 16, 8]

    stage = int(layer / 3)
    for index in range(0, layer, 1):
        if index == 0:  # first conv layer before block
            flop += channel[0] * width[0] * width[0] * 9 * 3 * prune_rate
        elif index in [1, 2]:  # first block of first stage
            flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate ** 2)
        elif 2 < index <= stage:  # other blocks of first stage
            if index % 2 != 0:
                # first layer of block, only output channal reduced, input channel remain the same
                flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate)
            elif index % 2 == 0:
                # second layer of block, both input and output channal reduced
                flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate ** 2)
        elif stage < index <= stage * 2:  # second stage
            if index % 2 != 0:
                flop += channel[1] * width[1] * width[1] * 9 * channel[1] * (prune_rate)
            elif index % 2 == 0:
                flop += channel[1] * width[1] * width[1] * 9 * channel[1] * (prune_rate ** 2)
        elif stage * 2 < index <= stage * 3:  # third stage
            if index % 2 != 0:
                flop += channel[2] * width[2] * width[2] * 9 * channel[2] * (prune_rate)
            elif index % 2 == 0:
                flop += channel[2] * width[2] * width[2] * 9 * channel[2] * (prune_rate ** 2)

    # offset for dimension change between blocks
    offset1 = channel[1] * width[1] * width[1] * 9 * channel[1] * prune_rate - channel[1] * width[1] * width[1] * 9 * \
            channel[0] * prune_rate
    offset2 = channel[2] * width[2] * width[2] * 9 * channel[2] * prune_rate - channel[2] * width[2] * width[2] * 9 * \
            channel[1] * prune_rate
    flop = flop - offset1 - offset2
    print(flop)
    return flop


def cal_res(layer, rate):
    flop_rate = 1 - int(cifar_resnet_flop(layer, rate)) / int(cifar_resnet_flop(layer, 1))
    print(flop_rate)
    return flop_rate

if __name__ == '__main__':
    cal_res(110, 0.6)
    cal_res(110, 0.5)

    # imagenet_flop(50, 1)
    # imagenet_flop(50, 0.9)
    # imagenet_flop(101, 1)
    # cal(imagenet_flop, 101, 0.7)
    # a=[]
    # for x in range(9, 0, -1):
    #     print("*" * 100,x)
    #     flop = round(cal_res(110, x / 10),5 )
    #     a.append(flop)
    # print(0,a)
    print(1)
