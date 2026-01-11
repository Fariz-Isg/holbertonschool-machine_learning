#!/usr/bin/env python3
"""
Plotting all 5 previous graphs in one figure
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Plots all 5 previous graphs in one figure
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('All in One')

    # Plot 1: y = x^3
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1.plot(np.arange(0, 11), y0, 'r-')
    ax1.set_xlim(0, 10)
    ax1.set_title('y0', fontsize='x-small')  # Usually internal names aren't titles, but complying with 'titles small'
    # Actually, previous plots didn't have titles for the first one?
    # Wait, the prompt says "All axis labels and plot titles should have a font size of x-small".
    # 0-line.py didn't have title. 1-scatter had "Men's Height vs Weight".
    # I should use the titles from previous tasks if they existed.
    # Task 0: No title. Task 1: "Men's Height vs Weight". Task 2: "Exponential Decay of C-14".
    # Task 3: "Exponential Decay of Radioactive Elements". Task 4: "Project A".
    # For Task 0, since no title was specified in Task 0, I might skip it or keep it empty. However, typical "All in one" tasks often implying copying exact previous styles.
    # BUT, looking at visual examples of this Holberton task (common online), typically the first one has no title? Or maybe it does?
    # Re-reading prompt: "All axis labels and plot titles should have a font size of x-small".
    # It implies wherever there IS a title/label.
    # I will stick to previous tasks' specifications.
    
    # Task 0 had no labels/title specified in its prompt.
    # Task 1: Title "Men's Height vs Weight", xlabel "Height (in)", ylabel "Weight (lbs)"
    # Task 2: Title "Exponential Decay of C-14", xlabel "Time (years)", ylabel "Fraction Remaining", scale log
    # Task 3: Title "Exponential Decay of Radioactive Elements", xlabel "Time (years)", ylabel "Fraction Remaining", legend
    # Task 4: Title "Project A", xlabel "Grades", ylabel "Number of Students"

    # Plot 1
    ax1.plot(np.arange(0, 11), y0, 'r-')
    ax1.set_xlim(0, 10)
    
    # Plot 2
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2.scatter(x1, y1, c='magenta', s=10) # s=10 is default roughly?
    ax2.set_xlabel('Height (in)', fontsize='x-small')
    ax2.set_ylabel('Weight (lbs)', fontsize='x-small')
    ax2.set_title("Men's Height vs Weight", fontsize='x-small')

    # Plot 3
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax3.plot(x2, y2)
    ax3.set_xlabel('Time (years)', fontsize='x-small')
    ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax3.set_title('Exponential Decay of C-14', fontsize='x-small')
    ax3.set_yscale('log')
    ax3.set_xlim(0, 28650)

    # Plot 4
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax4.plot(x3, y31, 'r--', label='C-14')
    ax4.plot(x3, y32, 'g-', label='Ra-226')
    ax4.set_xlabel('Time (years)', fontsize='x-small')
    ax4.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax4.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    ax4.set_xlim(0, 20000)
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper right', fontsize='x-small')

    # Plot 5
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax5.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    ax5.set_xlabel('Grades', fontsize='x-small')
    ax5.set_ylabel('Number of Students', fontsize='x-small')
    ax5.set_title('Project A', fontsize='x-small')
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()
