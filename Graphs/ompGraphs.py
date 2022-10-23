import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.ticker as plticker

colors = []
def barColor(attempt):
    if "N" in attempt:
        colors.append('red')
    elif "S" in attempt:
        colors.append('green')
    elif "D" in attempt:
        colors.append('purple')
    else:
        colors.append('blue')

def scheduleBar(df, title):
    new = df.apply(
        lambda row: barColor(row['Attempt']),
        axis=1)

    fig, axis = plt.subplots(nrows=2, ncols=2)
    ax1 = df.plot.bar(x='Attempt', y='1st_Speedup', rot=30,ax=axis[0][0],color=colors)
    ax1.title.set_text('A: 512 x 32, B: 32 x 512, C: 512 x 512')
    ax1.set_xlabel('')
    ax1.set_ylabel('Speedup')
    ax1.set_ylim([0, 30])
    ax2 = df.plot.bar(x='Attempt', y='2nd_Speedup', rot=30,ax=axis[0][1],color=colors)
    ax2.title.set_text('A: 1024 x 32, B: 32 x 1024, C: 1024 x 1024')
    ax2.set_xlabel('')
    ax2.set_ylabel('Speedup')
    ax2.set_ylim([0, 30])
    ax3 = df.plot.bar(x='Attempt', y='3rd_Speedup', rot=30,ax=axis[1][0],color=colors)
    ax3.title.set_text('A: 1024 x 64, B: 64 x 1024, C: 1024 x 1024')
    ax3.set_xlabel('')
    ax3.set_ylabel('Speedup')
    ax3.set_xlabel('Schedule(Thread Count)')
    ax3.set_ylim([0, 30])
    ax4 = df.plot.bar(x='Attempt', y='4th_Speedup', rot=30,ax=axis[1][1], color=colors)
    ax4.title.set_text('A: 2048 x 64, B: 64 x 2048, C: 2048 x 2048')
    ax4.set_xlabel('')
    ax4.set_ylabel('Speedup')
    ax4.set_xlabel('Schedule(Thread Count)')
    ax4.set_ylim([0, 30])

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    red_patch = mpatches.Patch(color='red', label='None')
    blue_patch = mpatches.Patch(color='blue', label='Guided')
    green_patch = mpatches.Patch(color='green', label='Static')
    purple_patch = mpatches.Patch(color='purple', label='Dynamic')
    fig.legend(handles=[red_patch, green_patch, purple_patch, blue_patch], title="Schedule Types")
    fig.suptitle(title, fontsize=16)
    plt.show()


def ompLine(df, title):
    loc = plticker.MultipleLocator(base=1.0)
    fig, axis = plt.subplots(nrows=4, ncols=2)
    dfNone = df[df['Schedule'] == "None"]
    dfStat = df[df['Schedule'] == "Static"]
    dfDyn = df[df['Schedule'] == "Dynamic"]
    dfGui = df[df['Schedule'] == "Guided"]

    ax1 = dfNone.plot.line(x='ThreadCt', y='smallArr', rot=30, ax=axis[0][0],style='.-',color='red')
    ax1.set_ylim([0, 50])
    ax1.get_legend().remove()
    ax1.xaxis.set_major_locator(loc)
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    ax1.title.set_text('A: 512 x 32, B: 32 x 512, C: 512 x 512')
    ax2 = dfNone.plot.line(x='ThreadCt', y='largeArr', rot=30, ax=axis[0][1],style='.-',color='red')
    ax2.set_ylim([0, 50])
    ax2.get_legend().remove()
    ax2.xaxis.set_major_locator(loc)
    ax2.set_xticklabels([])
    ax2.set_xlabel('')
    ax2.title.set_text('A: 2048 x 64, B: 64 x 2048, C: 2048 x 2048')
    ax3 = dfStat.plot.line(x='ThreadCt', y='smallArr', rot=30, ax=axis[1][0], style='.-', color='green')
    ax3.set_ylim([0, 50])
    ax3.get_legend().remove()
    ax3.xaxis.set_major_locator(loc)
    ax3.set_xticklabels([])
    ax3.set_xlabel('')
    ax4 = dfStat.plot.line(x='ThreadCt', y='largeArr', rot=30, ax=axis[1][1], style='.-', color='green')
    ax4.set_ylim([0, 50])
    ax4.get_legend().remove()
    ax4.xaxis.set_major_locator(loc)
    ax4.set_xticklabels([])
    ax4.set_xlabel('')
    ax5 = dfDyn.plot.line(x='ThreadCt', y='smallArr', rot=30, ax=axis[2][0], style='.-', color='purple')
    ax5.set_ylim([0, 50])
    ax5.get_legend().remove()
    ax5.xaxis.set_major_locator(loc)
    ax5.set_xticklabels([])
    ax5.set_xlabel('')
    ax6 = dfDyn.plot.line(x='ThreadCt', y='largeArr', rot=30, ax=axis[2][1], style='.-', color='purple')
    ax6.set_ylim([0, 50])
    ax6.get_legend().remove()
    ax6.xaxis.set_major_locator(loc)
    ax6.set_xticklabels([])
    ax6.set_xlabel('')
    ax7 = dfGui.plot.line(x='ThreadCt', y='smallArr', rot=30, ax=axis[3][0], style='.-', color='blue')
    ax7.set_ylim([0, 50])
    ax7.get_legend().remove()
    ax7.set_xlabel('Thread Count')
    ax7.set_ylabel('Speedup')
    ax7.xaxis.set_major_locator(loc)
    ax8 = dfGui.plot.line(x='ThreadCt', y='largeArr', rot=30, ax=axis[3][1], style='.-', color='blue')
    ax8.set_ylim([0, 50])
    ax8.get_legend().remove()
    ax8.set_xlabel('Thread Count')
    ax8.set_ylabel('Speedup')
    ax8.xaxis.set_major_locator(loc)

    red_patch = mpatches.Patch(color='red', label='None')
    blue_patch = mpatches.Patch(color='blue', label='Guided')
    green_patch = mpatches.Patch(color='green', label='Static')
    purple_patch = mpatches.Patch(color='purple', label='Dynamic')
    fig.legend(handles=[red_patch, green_patch, purple_patch, blue_patch], title="Schedule Types")
    fig.suptitle("Effects of Thread Count and Schedule Type on Speedup with " + title, fontsize=16)

    plt.show()

def cudaGraph(df):
    fig, axis = plt.subplots(nrows=4, ncols=2)
    df64 = df[df['matrixSz'] == "64x64"]
    df256 = df[df['matrixSz'] == "256x256"]
    df1024 = df[df['matrixSz'] == "1024x1024"]
    df2048 = df[df['matrixSz'] == "2048x2048"]
    ax1 = df64.plot.line(x='tileSz', y='NaiveSpeedup', rot=0, ax=axis[0][0], style='.-', color='blue')
    ax1.get_legend().remove()
    ax1.set_ylim([0, 5])
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    ax1.title.set_text("Naive\n64x64")
    ax2 = df64.plot.line(x='tileSz', y='TiledSpeedup', rot=0, ax=axis[0][1], style='.-', color='purple')
    ax2.get_legend().remove()
    ax2.set_ylim([0, 5])
    ax2.set_xticklabels([])
    ax2.set_xlabel('')
    ax2.title.set_text("Tiled\n64x64")
    ax3 = df256.plot.line(x='tileSz', y='NaiveSpeedup', rot=0, ax=axis[1][0], style='.-', color='blue')
    ax3.get_legend().remove()
    ax3.set_ylim([0, 60])
    ax3.set_xticklabels([])
    ax3.set_xlabel('')
    ax3.title.set_text("256x256")
    ax4 = df256.plot.line(x='tileSz', y='TiledSpeedup', rot=0, ax=axis[1][1], style='.-', color='purple')
    ax4.get_legend().remove()
    ax4.set_ylim([0, 60])
    ax4.set_xticklabels([])
    ax4.set_xlabel('')
    ax4.title.set_text("256x256")
    ax5 = df1024.plot.line(x='tileSz', y='NaiveSpeedup', rot=0, ax=axis[2][0], style='.-', color='blue')
    ax5.get_legend().remove()
    ax5.set_ylim([0, 400])
    ax5.set_xticklabels([])
    ax5.set_xlabel('')
    ax5.title.set_text("1024x1024")
    ax6 = df1024.plot.line(x='tileSz', y='TiledSpeedup', rot=0, ax=axis[2][1], style='.-', color='purple')
    ax6.get_legend().remove()
    ax6.set_ylim([0, 400])
    ax6.set_xticklabels([])
    ax6.set_xlabel('')
    ax6.title.set_text("1024x1024")
    ax7 = df2048.plot.line(x='tileSz', y='NaiveSpeedup', rot=0, ax=axis[3][0], style='.-', color='blue')
    ax7.get_legend().remove()
    ax7.set_ylim([0, 2000])
    ax7.title.set_text("2048x2048")
    ax7.set_xlabel('Tile Size')
    ax7.set_ylabel('Speedup')
    ax8 = df2048.plot.line(x='tileSz', y='TiledSpeedup', rot=0, ax=axis[3][1], style='.-', color='purple')
    ax8.get_legend().remove()
    ax8.set_ylim([0, 2000])
    ax8.title.set_text("2048x2048")
    ax8.set_xlabel('Tile Size')
    ax8.set_ylabel('Speedup')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    fig.suptitle("Effects of Tile Size and Matrix Size on Speedup with Naive Cuda and Tiled Cuda", fontsize=16)

    plt.show()

def mainGraph(df):
    colorsArr = ['green', 'red', 'blue', 'purple']
    df256 = df[df['matrixSz'] == "256x256"]
    df1024 = df[df['matrixSz'] == "1024x1024"]
    df64 = df[df['matrixSz'] == "64x64"]
    df2048 = df[df['matrixSz'] == "2048x2048"]
    fig, axis = plt.subplots(nrows=4, ncols=1)
    ax1 = df256.plot.bar(x='method', y='speedup', rot=0, ax=axis[1], style='.-', color=colorsArr)
    ax1.get_legend().remove()
    ax1.title.set_text("Matrix Size: 256 x 256")
    ax1.set_ylim([0, 100])
    ax1.set_xlabel('')
    ax1.set_ylabel('Speedup')
    ax2 = df1024.plot.bar(x='method', y='speedup', rot=0, ax=axis[2], style='.-', color=colorsArr)
    ax2.get_legend().remove()
    ax2.title.set_text("Matrix Size: 1024 x 1024")
    ax2.set_ylim([0, 400])
    ax2.set_xlabel('')
    ax2.set_ylabel('Speedup')
    ax3 = df64.plot.bar(x='method', y='speedup', rot=0, ax=axis[0], style='.-', color=colorsArr)
    ax3.get_legend().remove()
    ax3.title.set_text("Matrix Size: 64 x 64")
    ax3.set_ylim([0, 5])
    ax3.set_xlabel('')
    ax3.set_ylabel('Speedup')
    ax4 = df2048.plot.bar(x='method', y='speedup', rot=0, ax=axis[3], style='.-', color=colorsArr)
    ax4.get_legend().remove()
    ax4.title.set_text("Matrix Size: 2048 x 2048")
    ax4.set_ylim([0, 2000])
    ax4.set_xlabel('')
    ax4.set_ylabel('Speedup')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    fig.suptitle("Speedups with Final Versions of OpenMP and CUDA Methods", fontsize=16)

    plt.show()

def main():
    dfBar = pd.read_csv('naiveOMP.csv')
    dfnLine = pd.read_csv('ompLine.csv')
    dfNtrans = pd.read_csv('naiveTrans.csv')
    dftLine = pd.read_csv('transposeLine.csv')
    dfCuda = pd.read_csv('cudaData.csv')
    dfTotal = pd.read_csv('mainGraph.csv')

    naivetitle = "Naive OpenMP to Naive Comparison"
    scheduleBar(dfBar, naivetitle)
    title = "Naive OpenMP"
    ompLine(dfnLine, title)
    transtitle = "Transpose OpenMP to Naive Comparison"
    scheduleBar(dfNtrans, transtitle)
    title = "Transpose OpenMP"
    ompLine(dftLine, title)
    cudaGraph(dfCuda)
    mainGraph(dfTotal)


if __name__ == '__main__':
    main()
