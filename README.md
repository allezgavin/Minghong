# Minghong
MFM quant model for summer internship at Minghong Investment.

backtest.py: A module for backtesting, given a portofolio.

factor_test.py: Contains a pool of factors. Calculates factor(s) and test if a factor is significant.

select_factors.py: Select factors with a loop, decolinearize them, and put them in a .csv file.

optimization.py: Calculate the optimal portfolio for each day with quadratic programming.

global_var.py: Sets global variables such as the stock pool, frequency of holding adjustments, database connection info, etc.

注：股票涨幅命名为'gain'。股票涨幅定义为chg / 100.

Notes on Aug 18, 2023
这两周参阅了更广泛的资料，并对模型的许多方面做出了改进。针对处理速度慢的问题，我尽量避免使用for循环，而是改用pandas内置的method, 例如rolling, cumprod等，速度显著提升。optimization采用multiprocessing并行计算。

factors.py文件合并到了factors_test.py中。factor的定义中舍弃了'lag'，并重新定义了诸多合成因子所需要的函数。计算过程中经常要将因子按日期或股票groupby，为了提高速度，我将对应的index以字典的形式存储为全局变量。

alpha因子和风格因子也进行了区分。风格因子全部进行行业和市值对数的中性化，alpha因子则无需统一中性化。在optimization中，针对alpha因子和风格因子的处理方式也有所不同，例如预测因子收益率的半衰期和二次规划中允许的最大因子暴露。下一步可以采用择时策略对风格因子进行轮动。

factor_test中删除了t_test, 转而使用IC和IR作为因子检验标准。结果显示，许多过去非常有效的因子在2022和2023年纷纷失效。其中，来自于WorldQuant的alpha因子可能是由于过拟合而失效。因子的参数还有待调整。

select_factor的方法有重大改变。alpha对所有收益率直接Lasso回归，剔除系数为0的因子；风格因子则是在横截面上（逐日）对收益率Lasso回归，剔除系数为0次数最多的因子。通过调整斜率惩罚项系数的值来获得目标数量的因子。选择完因子后，factor_regression_history使用Ridge回归，以确保因子收益率的回归值较为稳定，尤其是当因子具有共线性时。

optimization中加入了许多限制条件，例如最大alpha因子和风格因子暴露，行业中性，市值中性等等。

Notes on Aug 6, 2023
本repository中的多因子模型主要参考了华泰基金的研报和一些知乎上的资料。程序先从Sequel数据库中下载market, finance, finance_deriv, company, indexprice, indexweight等表中的相关数据，再利用pandas处理数据。此方法似乎运行速度较慢，可能和文件过大有关，也可能是我写的程序的原因。我在考虑使用一个移动硬盘，配合数据库软件来存取计算过程中产生的数据（例如各因子的数值）。

factor_test.py文件中的calc_factors()函数将market, finance, finance_deriv和company四个table合为一个dataframe，剔除空值，计算各因子的值，然后对因子进行分行业的标准化，这样可以做到行业中性。相比于在优化模型中添加行业中性的约束条件，这样似乎更方便计算。随后，我讲所有的因子对市值进行回归并取残差，作为最终的因子值，以做到市值中性（异方差性会不会对结果有所影响？）。处理过后的数据存储到factors.csv文件。
要计算的因子存储在factors.py中。存储的形式为字典，其中'indicators'代表计算此factor需要用到的数据的名称，'function'是计算的函数，例如市盈率的indicators是股价和每股收益，函数是相除。'lag'可以用来应对不在同一行的数据，例如relative_strength_1m要将收盘价和1个月前的收盘价进行对比，因此indicators为['close', 'close']，lag为21行（每月约有21个交易日）。对于volatility factors和momentum factors, 我进行了一点灵活的运用，但是代价是运行速度很慢。不知道有没有更好的计算factor的写法？
factor_test.py中的t_test()和group_backtest()函数可以对单个因子进行t检验和分层回测，方便挖掘因子。目前绝大部份因子每日的t值都不显著大于0或小于0，说明没有alpha因子。
select_factors.py用来选取factors.csv文件中的因子，将T日的因子值和T+1日的股票涨幅进行回归，力求选取在过去一年的数据上平均adjusted R^2最大的因子组合。我使用了一个循环，每次选取对adjusted R^2增加最大的一个因子，直到adjusted R^2无法增加。
因子选取完毕后，每个因子对前面所有的因子回归、取残差，作为新的因子值，存储到selected_factors.csv文件中。这步是为了去除因子的共线性。这是我在一个知乎视频上看到的方法，但这个“偷懒”的方法带来了一些问题：新的因子值的现实意义不明，无法通过现实意义来约束alpha因子收益率的正负性，也无法得到一个只在特定因子上暴露套利的投资组合。我可能会改掉这个程序，用主成分分析或者别的方法选取因子。
select_factors.py的factor_regression_history()函数用selected_factors.csv中的因子值，对各因子的收益率做横截面上（逐日）的回归，输出到factor_return.csv文件中，同时将每只股票每日的残差输出到residual.csv文件，方便之后计算投资组合的风险结构。
目前预测T+1日因子收益率的方式是过去所有的因子收益率的平均值（见optimization.py中的expected_f）。这个方法感觉不是很好，也许可以尝试使用半衰期等方法。不过目前来说所有的因子收益率波动都很大，暂时看不出效果，就当一个placeholder先放着了。
optimization.py使用cvxopt的二次规划，算出最佳的投资组合。目标函数为风险调整后的收益率，约束为1.每个因子上的暴露的绝对值不能超过x_k；2.不能做空；3.全仓。optimization_scipy.py是使用scipy.minimize的优化模型，运行速度奇慢，不适用。
为了方便回测，optimize()函数支持传入日期参数，计算T+1日的投资组合。backtest_portfolio()函数通过不断调用optimize()函数，获得历史投资组合，写入backtest_portfolio.csv文件。
将backtest_portfolio.csv传入backtest()，即可得到回测结果。
questions.txt记录了目前的一些疑问。模型应该还是非常粗糙的，很多重要的问题我也没有考虑，例如线性回归的异方差性、对空值的处理、程序的运行速度等。
