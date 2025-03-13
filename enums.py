from enum import Enum

class Side(Enum):
    NONE = None
    LEFT = "left"
    RIGHT = "right"

class ExoState(Enum):
    STOP = 0
    OPEN = 1
    CLOSE = 2

    START = 3
    LOCK = 4
    READY = 6

    HIDE_STOP = 7
    HIDE_OPEN = 8
    HIDE_CLOSE = 9

class Cue(Enum):
    EMPTY: int = 0
    CLOSE: int = 1
    RELAX: int = 2
    STARTIN5: int = 3
    END: int = 4
    HOVLEFT: int = 5
    HOVRIGHT: int = 6
    
    CLOSE_LEFT: int = 7
    CLOSE_RIGHT: int = 8
    HOVLEFT_LONG: int = 9
    HOVRIGHT_LONG: int = 10

    EXO_INACTIVE = 1000000000
    EXO_ACTIVE = 1255255255
    EXO_READY = 1000255000
    EXO_BLOCKED = 1255000000

DisplayText: dict = {
    Cue.EMPTY.value: '',
    Cue.CLOSE.value: 'Hand Schließen',
    Cue.RELAX.value: '',
    Cue.CLOSE_LEFT.value: 'close left',
    Cue.CLOSE_RIGHT.value: 'close right',

    Cue.HOVLEFT.value: '<<<',
    Cue.HOVRIGHT.value: '>>>',
    Cue.HOVLEFT_LONG.value: '<<<<<<',
    Cue.HOVRIGHT_LONG.value: '>>>>>>',

    Cue.STARTIN5.value: 'Start in 5 Sekunden',
    Cue.END.value: 'ENDE'
}
