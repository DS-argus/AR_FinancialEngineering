import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=3, suppress=True)


class BinomialPricing:
    def __init__(self,
                 S0: float,
                 T: float,
                 Periods: int,
                 Volatility: float,
                 Risk_free_rate: float,
                 Dividend: float = 0.0):

        """

        :param S0: 초기 주가
        :param T: 만기(연)
        :param Periods: 구간
        :param Volatility: 변동성(연)
        :param Risk_free_rate: 무위험이자율(연)
        :param Dividend: 배당수익률(연)
        """
        self.__S0 = S0
        self.__T = T
        self.__P = Periods
        self.__V = Volatility
        self.__R = Risk_free_rate
        self.__D = Dividend

        """
        self.__u = black-scholes parameters로 변환한 주가 상승률
        self.__d = black-scholes parameters로 변환한 주가 하락률
        self.__q = black-scholes parameters로 변환한 주가가 상승할 위험중립확률
        """
        self.__u = np.exp(self.__V * np.sqrt(self.__T / self.__P))
        self.__d = 1 / self.__u

        if (1 + self.__R >= self.__u + self.__D) or (1 + self.__R <= self.__d + self.__D):
            raise Exception("Type B arbitrage exists,\n please modify your inputs")

        self.__q = (np.exp((self.__R - self.__D) * self.__T / self.__P) - self.__d) / (self.__u - self.__d)

    @property
    def getparams(self):
        return {
            "S0": self.__S0,
            "T": self.__T,
            "P": self.__P,
            "V": self.__V,
            "R": self.__R,
            "D": self.__D,
            "u": self.__u,
            "d": self.__d,
            "q": self.__q
        }

    def stock_lattice(self) -> np.array:

        stock_lattice = np.zeros((self.__P + 1)**2).reshape([self.__P + 1, self.__P + 1])

        for i in range(self.__P + 1):
            for j in range(self.__P - i, self.__P + 1):
                value = self.__S0 * self.__u ** (self.__P - i) * self.__d ** (i + j - self.__P)
                stock_lattice[i][j] = value

        return stock_lattice

    def future_lattice(self) -> np.array:

        future_lattice = np.zeros((self.__P + 1)**2).reshape([self.__P + 1, self.__P + 1])

        for i in range(self.__P + 1):
            future_lattice[i][self.__P] = self.stock_lattice()[i][self.__P]

        for i in range(1, self.__P + 1):
            for j in range(self.__P - 1, self.__P - i - 1, -1):
                future_lattice[i][j] = (self.__q * future_lattice[i - 1][j + 1] +
                                        (1 - self.__q) * future_lattice[i][j + 1])

        return future_lattice

    def option_lattice(self,
                       underlying_lattice: np.array,
                       K: float,
                       option_type: str,
                       exercise_type: str) -> tuple:
        """

        :param underlying_lattice: 기초자산으로 사용할 격자
        :param K: 행사가격
        :param option_type: call or put
        :param exercise_type: european or american
        :return: (옵션 가격 격자, 옵션 가격 at t=0, 조기행사 발생하는 t)
        """
        assert option_type == "c" or option_type == "p"
        assert exercise_type == "european" or exercise_type == "american"

        if option_type == "c":
            sign = 1
        else:
            sign = -1

        early_exercise = []

        periods = len(underlying_lattice) - 1
        option_lattice = np.zeros((periods + 1)**2).reshape([periods + 1, periods + 1])

        for i in range(periods + 1):
            option_lattice[i][periods] = max(sign * (underlying_lattice[i][periods] - K), 0)

        for i in range(1, periods + 1):

            # backward로 계산해야하는 것 주의
            for j in range(periods - 1, periods - i - 1, -1):

                if exercise_type == "european":
                    option_lattice[i][j] = (self.__q * option_lattice[i - 1][j + 1] +
                                            (1 - self.__q) * option_lattice[i][j + 1]) * \
                                           np.exp(-self.__R * self.__T / self.__P)

                elif exercise_type == "american":
                    option_lattice[i][j] = max(max(sign * (underlying_lattice[i][j] - K), 0),
                                               (self.__q * option_lattice[i - 1][j + 1] +
                                                (1 - self.__q) * option_lattice[i][j + 1]) *
                                               np.exp(-self.__R * self.__T / self.__P))

                    # 조기행사 되는 t를 리스트에 저장
                    if option_lattice[i][j] == max(sign * (underlying_lattice[i][j] - K), 0):
                        early_exercise.append(j)

        return option_lattice, option_lattice[periods, 0], early_exercise


if __name__ == "__main__":
    bn = BinomialPricing(100, 0.25, 15, 0.3, 0.02, 0.01)                    # binomial 객체 생성

    stock_path = bn.stock_lattice()                                         # 주가 격자 생성
    future_path = bn.future_lattice()                                       # futures 격자 생성(by risk_neutral_pricing)

    # 1. 주가를 기초자산으로 하는 옵션
    underlying_path1 = stock_path

    # 2. futures를 기초자산으로 하는 옵션(option, futures 만기 일치)
    underlying_path2 = future_path

    # 3. option 만기가 futures 만기보다 짧을 때 (option 만기는 n = 10, futures 만기는 n = 15)
    underlying_path3 = future_path[5:, :11]

    result = bn.option_lattice(underlying_path3, 110, "c", "american")       # binomial pricing 실행

    print(bn.getparams)                                                     # parameters
    print(underlying_path3)                                                 # underlying path
    print(result[0])                                                        # lattices
    print(f"price of option : {result[1]}")                                 # price
    print(f"earliest time to early exercise : time = {min(result[2])}")     # early exercise
