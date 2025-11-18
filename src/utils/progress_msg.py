import time
import datetime


class ProgressMsg:
    def __init__(self, max_iter, min_time_interval=0.5):
        """
        Args:
            max_iter : (max_epoch, max_data_length, ...)
            min_time_interval (second)
        """
        self.current_iter = None
        self.start_iter = None
        self.max_iter = max_iter
        self.min_time_interval = min_time_interval
        self.start_time = time.time()
        self.progress_time = self.start_time

    def start(self, start_iter):
        assert len(self.max_iter) == len(start_iter), 'start_iter should have same length with max variable.'

        self.start_iter = start_iter
        self.current_iter = start_iter
        self.start_time = time.time()
        self.progress_time = self.start_time

    def calculate_progress(self, current_iter):
        self.progress_time = time.time()
        assert len(self.max_iter) == len(current_iter), 'current should have same length with max variable.'

        for i in range(len(self.max_iter)):
            assert current_iter[i] <= self.max_iter[i], 'current value should be less than max value.'

        start_per = 0
        for i in reversed(range(len(self.max_iter))):
            start_per += self.start_iter[i]
            start_per /= self.max_iter[i]
        start_per *= 100

        pg_per = 0
        for i in reversed(range(len(self.max_iter))):
            pg_per += current_iter[i]
            pg_per /= self.max_iter[i]
        pg_per *= 100

        pg_per = (pg_per-start_per) / (100-start_per) * 100

        if pg_per != 0:
            elapsed = time.time() - self.start_time
            total = 100*elapsed/pg_per
            remain = total - elapsed
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            remain_str = str(datetime.timedelta(seconds=int(remain)))
            total_str = str(datetime.timedelta(seconds=int(total)))
        else:
            elapsed = time.time() - self.start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            remain_str = 'INF'
            total_str = 'INF'

        return pg_per, elapsed_str, remain_str, total_str

    def print_prog_msg(self, current_iter):
        if time.time() - self.progress_time >= self.min_time_interval:
            pg_per, elapsed_str, remain_str, total_str = self.calculate_progress(current_iter)
            txt = (
                f" \033[K>>> progress : {pg_per:.2f}%, "
                f"elapsed: {elapsed_str}, "
                f"remaining: {remain_str}, "
                f"total: {total_str}      "
            )
            print(txt, end='\r', flush=True)
            return txt.replace('\t', '')
        return

    @staticmethod
    def get_start_msg():
        return 'Start >>>'

    def get_finish_msg(self):
        total = time.time() - self.start_time
        total_str = str(datetime.timedelta(seconds=int(total)))
        txt = 'Finish >>> (total elapsed time : %s)' % total_str
        return txt
