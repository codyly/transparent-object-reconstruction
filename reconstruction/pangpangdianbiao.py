# pangpangdianbiao.py

from tqdm import trange

# 第一档：电量为年用电量2760千瓦以下部分，电价不作调整，仍为每千瓦时0.538元；
# 第二档：电量为2761至4800千瓦时部分，电价在第一档电价基础上加价0.05元，为每千瓦时0.588元；
# 第三档：电量超过4800千瓦时部分，电价在第一档电价基础上加价0.3元，为每千瓦时0.838元。
# 对于选择峰谷分时电价的杭州市居民用户，按高峰、低谷合计电量执行阶梯电价，
# 其中第一档电量峰、谷电价仍按每千瓦时0.568元、0.288元执行，第二、三档电量峰、谷电价在第一档电价基础上均同步加价0.05元、0.3元。

class pangpang:
    def __init__(self, daily_feng_use=8, daily_gu_use=2):
        self.fpd = daily_feng_use
        self.gpd = daily_gu_use

class fg_meter:
    def __init__(self):
        self.stages = []
        self.prices = []
        self.thresholds = [2760, 4800, float("inf")]
        for i in range(3):
            self.stages.append({'F':0,'G':0})
            self.prices.append({'F':0.568 + 0.05 * (i > 0),'G':0.288 + 0.3 * (i > 0)})
        self.total = 0
        self.cost = 0

    def use_feng(self, power):
        stage_id = 0
        while power > 0 and stage_id < 3:
            if self.total <= self.thresholds[stage_id]:  # first stage
                diff = self.thresholds[stage_id] - self.total
                if power <= diff:
                    self.total += power
                    self.stages[stage_id]['F'] += power
                else:
                    self.total = self.thresholds[stage_id]
                    self.stages[stage_id]['F'] += diff;
                    power = power - diff
            stage_id += 1
        self.cal_cost()
        return self.total

    def use_gu(self, power):
        stage_id = 0
        while power > 0 and stage_id < 3:
            if self.total <= self.thresholds[stage_id]:  # first stage
                diff = self.thresholds[stage_id] - self.total
                if power <= diff:
                    self.total += power
                    self.stages[stage_id]['G'] += power
                else:
                    self.total = self.thresholds[stage_id]
                    self.stages[stage_id]['G'] += diff;
                    power = power - diff
            stage_id += 1
        self.cal_cost()
        return self.total

    def cal_cost(self):
        self.cost = 0
        for i in range(3):
            self.cost = self.cost + self.stages[i]['F'] * self.prices[i]['F'] + self.stages[i]['G'] * self.prices[i]['G']
        return self.cost

    def simulation(self, daily_use, days):
        for i in trange(days):
            self.use_feng(daily_use['F'])
            self.use_gu(daily_use['G'])
        self.cal_cost()
        return self.cost

if __name__=='__main__':
    pp = pangpang()
    meter = fg_meter()
    cost = meter.simulation({'F':pp.fpd, 'G':pp.gpd}, 30)
    print(meter.total, cost)