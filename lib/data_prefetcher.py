import torch

class DataPrefetcher(object):
    def __init__(self, loader, stage):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload(stage)


    def preload(self, stage):
        try:
            if stage ==  'stage1':
                self.next_input, self.next_target, _, _ = next(self.loader, stage)
            elif stage == 'stage2':
                self.next_input, self.next_target, self.next_Sinput, self.next_Starget, self.label, _, _ = next(self.loader, stage)
            elif stage == 'stage3':
                self.next_input, self.next_target, _, _ = next(self.loader, stage)
        except:
            if stage == 'stage1':
                self.next_input = None
                self.next_target = None
            elif stage == 'stage2':
                self.next_input = None
                self.next_target = None
                self.next_Sinput = None
                self.next_Starget = None
                self.label = None
            elif stage == 'stage3':
                self.next_input = None
                self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            if stage == 'stage1':
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)
                self.next_input = self.next_input.float() #if need
                self.next_target = self.next_target.float() #if need
            elif stage == 'stage2':
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)
                self.next_Sinput = self.next_Sinput.cuda(non_blocking=True)
                self.next_Starget = self.next_Starget.cuda(non_blocking=True)
                self.label = self.label.cuda(non_blocking=True)
                self.next_input = self.next_input.float()  # if need
                self.next_target = self.next_target.float()  # if need
                self.next_Sinput = self.next_Sinput.float()  # if need
                self.next_Starget = self.next_Starget.float()  # if need
                self.label = self.label.float()  # if need
            elif stage == 'stage3':
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)
                self.next_input = self.next_input.float() #if need
                self.next_target = self.next_target.float() #if need

    def next(self, stage):
        torch.cuda.current_stream().wait_stream(self.stream)
        if stage == 'stage1':
            input = self.next_input
            target = self.next_target
            self.preload(stage)
            return input, target
        elif stage == 'stage2':
            input = self.next_input
            target = self.next_target
            Sinput = self.next_Sinput
            Starget = self.next_Starget
            label = self.label
            self.preload(stage)
            return input, target, Sinput, Starget, label
        elif stage == 'stage3':
            input = self.next_input
            target = self.next_target
            self.preload(stage)
            return input, target


