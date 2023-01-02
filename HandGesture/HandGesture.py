from Commons.landmarks import Hand
from Commons.vector import calcDistance2D, calcDotProduct, calcVector2D

class HandGesture():
    def __init__(self, *, button0: bool, button1: bool, scroll: bool, relative: bool):
        self.button0 = button0
        self.button1 = button1
        self.scroll = scroll
        self.relative = relative

    def __repr__(self):
        return f"button0: {self.button0}, button1: {self.button1}, scroll: {self.scroll}, relative: {self.relative}"

def handGestureFront(hand: Hand):
    #threshold = max(calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[9]), calcDistance2D(hand.raw_landmark[2], hand.raw_landmark[3])) * 1.3
    threshold = max(calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[9]), calcDistance2D(hand.raw_landmark[2], hand.raw_landmark[3])) * 1.7
    thumb_tip = hand.raw_landmark[4]
    index_tip = hand.raw_landmark[8]
    middle_tip = hand.raw_landmark[12]
    index_touch = calcDistance2D(thumb_tip, index_tip) / threshold < 1
    middle_touch = calcDistance2D(thumb_tip, middle_tip) / threshold < 1

    index_vector = calcVector2D(hand.raw_landmark[7], hand.raw_landmark[6])
    index_hand_vector = calcVector2D(hand.raw_landmark[0], hand.raw_landmark[5])
    index_close = calcDotProduct(index_vector, index_hand_vector) < 50

    middle_vector = calcVector2D(hand.raw_landmark[11], hand.raw_landmark[10])
    middle_hand_vector = calcVector2D(hand.raw_landmark[0], hand.raw_landmark[9])
    middle_close = calcDotProduct(middle_vector, middle_hand_vector) < 50

    pinky_vector = calcVector2D(hand.raw_landmark[19], hand.raw_landmark[18])
    pinky_hand_vector = calcVector2D(hand.raw_landmark[0], hand.raw_landmark[17])
    pinky_close = calcDotProduct(pinky_vector, pinky_hand_vector) < 50

    scroll = not index_touch and not middle_touch and index_close and middle_close and pinky_close

    return HandGesture(
        button0 = not scroll and index_touch,
        button1 = not scroll and middle_touch,
        scroll = scroll,
        relative = not scroll and pinky_close
    )

def handGestureSide(hand: Hand):
    #threshold = max(calcDistance2D(hand.raw_landmark[1], hand.raw_landmark[2]), calcDistance2D(hand.raw_landmark[2], hand.raw_landmark[3]))
    threshold = max(calcDistance2D(hand.raw_landmark[1], hand.raw_landmark[2]), calcDistance2D(hand.raw_landmark[2], hand.raw_landmark[3])) * 1.5
    thumb_tip = hand.raw_landmark[4]
    index_tip = hand.raw_landmark[8]
    middle_tip = hand.raw_landmark[12]
    index_touch = calcDistance2D(thumb_tip, index_tip) / threshold < 1
    middle_touch = calcDistance2D(thumb_tip, middle_tip) / threshold < 1

    index_vector = calcVector2D(hand.raw_landmark[7], hand.raw_landmark[6])
    index_hand_vector = calcVector2D(hand.raw_landmark[0], hand.raw_landmark[5])
    index_close = calcDotProduct(index_vector, index_hand_vector) < 40

    middle_vector = calcVector2D(hand.raw_landmark[11], hand.raw_landmark[10])
    middle_hand_vector = calcVector2D(hand.raw_landmark[0], hand.raw_landmark[9])
    middle_close = calcDotProduct(middle_vector, middle_hand_vector) < 35

    pinky_vector = calcVector2D(hand.raw_landmark[19], hand.raw_landmark[18])
    pinky_hand_vector = calcVector2D(hand.raw_landmark[0], hand.raw_landmark[17])
    pinky_close = calcDotProduct(pinky_vector, pinky_hand_vector) < 80
    #print(calcDotProduct(pinky_vector, pinky_hand_vector))

    scroll = index_close and middle_close and pinky_close

    return HandGesture(
        button0 = not scroll and index_touch,
        button1 = not scroll and middle_touch and not index_touch,
        scroll = scroll,
        relative = not scroll and pinky_close
    )

def handGestureTop(hand: Hand):
    index_threshold = max(calcDistance2D(hand.raw_landmark[8], hand.raw_landmark[7]), calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[9])) * 1.5
    middle_threshold = max(calcDistance2D(hand.raw_landmark[12], hand.raw_landmark[11]), calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[9])) * 1.8
    thumb_tip = hand.raw_landmark[4]
    index_tip = hand.raw_landmark[8]
    middle_tip = hand.raw_landmark[12]
    index_touch = calcDistance2D(thumb_tip, index_tip) / index_threshold < 1
    middle_touch = calcDistance2D(thumb_tip, middle_tip) / middle_threshold < 1
    #print(index_touch, middle_touch)
    #print(calcDotProduct(calcVector2D(hand.raw_landmark[5], hand.raw_landmark[9]), calcVector2D(hand.raw_landmark[4], hand.raw_landmark[3])))

    index_vector = calcVector2D(hand.raw_landmark[8], hand.raw_landmark[7])
    index_hand_vector = calcVector2D(hand.raw_landmark[6], hand.raw_landmark[5])
    index_close = calcDotProduct(index_vector, index_hand_vector) > 80
    #print(calcDotProduct(index_vector, index_hand_vector))

    middle_vector = calcVector2D(hand.raw_landmark[12], hand.raw_landmark[11])
    middle_hand_vector = calcVector2D(hand.raw_landmark[10], hand.raw_landmark[9])
    middle_close = calcDotProduct(middle_vector, middle_hand_vector) > 80
    #print(middle_close)

    pinky_vector = calcVector2D(hand.raw_landmark[20], hand.raw_landmark[19])
    pinky_hand_vector = calcVector2D(hand.raw_landmark[18], hand.raw_landmark[17])
    pinky_close = calcDotProduct(pinky_vector, pinky_hand_vector) > 100
    pinky_vector = calcVector2D(hand.raw_landmark[18], hand.raw_landmark[17])
    pinky_hand_vector = calcVector2D(hand.raw_landmark[17], hand.raw_landmark[0])
    pinky_close = pinky_close or calcDotProduct(pinky_vector, pinky_hand_vector) > 80
    pinky_close = pinky_close or calcDistance2D(hand.raw_landmark[19], hand.raw_landmark[18]) / calcDistance2D(hand.raw_landmark[18], hand.raw_landmark[17]) < 0.6
    #print(pinky_close)

    scroll = calcDotProduct(calcVector2D(hand.raw_landmark[5], hand.raw_landmark[9]), calcVector2D(hand.raw_landmark[4], hand.raw_landmark[3])) < 50
    scroll = scroll and index_close and middle_close and pinky_close

    return HandGesture(
        button0 = not scroll and index_touch,
        button1 = not scroll and middle_touch and not index_touch,
        scroll = scroll,
        relative = not scroll and pinky_close
    )

handGesture = {
    "Front": handGestureFront,
    "Side": handGestureSide,
    "Top": handGestureTop
}

def switchDirection(hand: Hand):
    if not hand:
        return None
    height_0 = calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[0])
    height_1 = calcDistance2D(hand.raw_landmark[17], hand.raw_landmark[0])
    height = max(height_0, height_1)
    width = calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[17])
    # print(f"{int(calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[0])*100)} {int(calcDistance2D(hand.raw_landmark[5], hand.raw_landmark[17])*100)} {int(calcDistance2D(hand.raw_landmark[17], hand.raw_landmark[0])*100)}")
    param = height/width
    dir = ""
    if param < 1.2:
        dir = "Top"
    elif param < 2.4:
        dir = "Front"
        #dir = None
    else:
        dir = "Side"
        #dir = None
    return dir

def handGestureRecognition(hand: Hand):
    dir = switchDirection(hand)
    if not dir:
        return None
    #print(dir)
    return handGesture[dir](hand)