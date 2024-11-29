# coding=utf-8
#
# THIS WORK HAS BEEN ADAPTED AND CHANGED!
# Original source code: https://github.com/zihangdai/xlnet
#
# Copyright 2024 Gabriel Lindenmaier
# Copyright 2019 XLNet Authors
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SPIECE_UNDERLINE = '▁'
SPIECE_NEWLINE = '<newline>'


def encode_pieces(sp_model, text, return_unicode=False, sample=False):
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)

    if len(pieces) > 1 and pieces[0] == SPIECE_UNDERLINE \
            and pieces[1] == SPIECE_NEWLINE:
        pieces = pieces[1:]

    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    return new_pieces
