'''from models.tacotron2 import Tacotron2Wave
model = Tacotron2Wave('pretrained/tacotron2_ar_adv.pth')
model = model.cuda()
wave = model.tts("اَلسَّلامُ عَلَيكُم يَا صَدِيقِي.")

wave_list = model.tts(["صِفر" ,"واحِد" ,"إِثنان", "ثَلاثَة" ,"أَربَعَة" ,"خَمسَة", "سِتَّة" ,"سَبعَة" ,"ثَمانِيَة", "تِسعَة" ,"عَشَرَة"])
'''

import torch
print(torch.cuda.is_available())