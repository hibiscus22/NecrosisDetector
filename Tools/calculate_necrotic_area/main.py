import cv2
import numpy as np
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


DICT_DAYS = {
    1: 7,
    2: 8,
    3: 9,
    4: 10,
    5: 12,
    6: 14,
    7: 15,
    8: 16,
    9: 17,
    10: 18
}    


def read_images_calculate_area_mask(cell_type:str, mask_type: str) -> pd.DataFrame:
    path_images = f"D:/TUW/images/{cell_type}/{mask_type}/"
    opt_imread = cv2.IMREAD_GRAYSCALE

    area_df = pd.DataFrame(columns = ['day', 'area'])

    for img_name in os.listdir(path_images):
        day_number = int(re.findall(r'\d+', img_name)[0])


        img = cv2.imread(path_images + img_name, opt_imread)/255
        
        area_df = pd.concat([area_df, pd.DataFrame([{'day': day_number, 
                    'area':np.sum(img) / np.size(img)}])], #To calculate the area we divide sum of pixels that are 1 / total of pixels
                     ignore_index = True)

    return area_df
        

def read_images_calculate_area(cell_line:str, dye_type: str) -> pd.DataFrame:
    path_images = f"D:/TUW/images/{cell_line}/{dye_type}/"
    opt_imread = cv2.IMREAD_GRAYSCALE

    area_df = pd.DataFrame(columns = ['day', 'area'])

    for img_name in os.listdir(path_images):
        day_number = int(re.findall(r'\d+', img_name)[0])


        img = cv2.imread(path_images + img_name, opt_imread)
        
        area_df = pd.concat([area_df, pd.DataFrame([{'day': day_number, 
                    'area':np.sum(img) / (np.size(img)*255)}])], #To calculate the area, total intensity / max possible intensity 
                     ignore_index = True)

    return area_df


def dead_alive_area_Pancreas(cell_line:str = "pancreas") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the number of pixels that are alive and dead acc. to the stain.
    Arguments:
        cell_line: str = Cell line that wants to be used. This algorithm works nice for pancreas, 
        because the masked necrotic area is small
    Returns:
        Two dataframes (alive and dead) with the respective area % per day    
    """
    path_images_all_cells = f"D:/TUW/images/{cell_line}/dapi_mask/"
    path_images_dead_cells = f"D:/TUW/images/{cell_line}/pi_mask/"
    opt_imread = cv2.IMREAD_GRAYSCALE

    area_alive = pd.DataFrame(columns = ['day', 'area'])
    area_dead = pd.DataFrame(columns = ['day', 'area'])

    for img_name in os.listdir(path_images_all_cells):
        day_number = int(re.findall(r'\d+', img_name)[0])


        img_all_cells = cv2.imread(path_images_all_cells + img_name, opt_imread)
        img_dead_cells = cv2.imread(path_images_dead_cells + img_name, opt_imread)
        
        spheroid_pixels = np.sum(img_all_cells)
        dead_pixels = np.sum(img_dead_cells)
        alive_pixels = spheroid_pixels - dead_pixels
        

        area_alive = pd.concat([area_alive, pd.DataFrame([{'day': day_number, 
                    'area': alive_pixels/ spheroid_pixels}])], #To calculate the area, total intensity / max possible intensity 
                     ignore_index = True)
        
        area_dead = pd.concat([area_dead, pd.DataFrame([{'day': day_number, 
                    'area':dead_pixels / spheroid_pixels}])], #To calculate the area, total intensity / max possible intensity 
                     ignore_index = True)

    return area_alive, area_dead

def dead_alive_area_CRC(cell_line:str = "ht29") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates dead area each day, returns dataframe.
    """

    path_images_all_cells = f"D:/TUW/images/{cell_line}/dapi/"
    path_images_dead_cells = f"D:/TUW/images/{cell_line}/pi/"
    opt_imread = cv2.IMREAD_GRAYSCALE

    area_alive = pd.DataFrame(columns = ['day', 'area'])
    area_dead = pd.DataFrame(columns = ['day', 'area'])

    for img_name in os.listdir(path_images_all_cells):
        day_number = int(re.findall(r'\d+', img_name)[0])

        img_all_cells = cv2.imread(path_images_all_cells + img_name, opt_imread)
        
        
        img_dead_cells = cv2.imread(path_images_dead_cells + img_name, opt_imread)
        
        # If the intensity of PI is higher than DAPI then weigh them together
        # if np.max(img_dead_cells) > np.max(img_all_cells):
        #     img_all_cells = (img_all_cells/np.max(img_all_cells)*255).astype(np.uint8)
        #     img_dead_cells = (img_dead_cells/np.max(img_dead_cells)*255).astype(np.uint8)
        
        #img_dead_cells = cv2.bitwise_and(img_dead_cells, img_all_cells)

        img_alive_cells = cv2.subtract(img_all_cells, img_dead_cells)

        #print(np.unique(img_alive_cells))

        # cv2.namedWindow("out", cv2.WINDOW_NORMAL)
        # cv2.imshow("out", np.concatenate([img_all_cells, img_dead_cells, img_alive_cells], axis = 1))
        # cv2.waitKey(0)
        
        dead_pixels = np.sum(img_dead_cells)
        alive_pixels = np.sum(img_alive_cells)
        spheroid_pixels = dead_pixels + alive_pixels
        
        # print(img_name, ":", spheroid_pixels, alive_pixels, dead_pixels)    

        area_alive = pd.concat([area_alive, pd.DataFrame([{'day': day_number, 
                    'area': alive_pixels/ spheroid_pixels}])], #To calculate the area, total intensity / max possible intensity 
                     ignore_index = True)
        
        area_dead = pd.concat([area_dead, pd.DataFrame([{'day': day_number, 
                    'area':dead_pixels / spheroid_pixels}])], #To calculate the area, total intensity / max possible intensity 
                     ignore_index = True)

    return area_alive, area_dead

def box_plot_progress(area_df: pd.DataFrame) -> None:
    area_df.replace({"day": DICT_DAYS}, inplace=True)
    sns.stripplot(data=area_df, x="day", y="area", jitter=True, alpha=0.6)
    # area_df['day'] = area_df['day']-1
    #sns.boxplot(data=area_df, x="day", y="area", hue="Origin")

def plot_progress(area_df: pd.DataFrame) -> None:
    #plt.figure(10, 6)
    area_df.replace({"day": DICT_DAYS}, inplace=True)
    plt.scatter(x=area_df['day'], y=area_df['area']*100, s=6)
    #plt.ylim([0, 0.03])
    plt.xticks(np.arange(min(area_df['day']), max(area_df['day'])+1, 1.0))
    plt.xlabel("Days")
    
    
    # if cell_line == "pancreas": plt.title("Hs766T") 
    # else: plt.title("HT-29")
    
def barchart(df: pd.DataFrame) -> None:
    # Calculate mean and standard deviation for each day
    
    daily_stats = df.groupby('day')['area'].agg(['mean', 'std'])

    # Create the bar chart
    plt.figure(figsize=(8, 6))

    # Bar for the mean value
    plt.bar(daily_stats.index, daily_stats['mean'], color='indianred', label='Necrotic')

    # Error bars 
    for day in daily_stats.index:
        mean = daily_stats.loc[day, 'mean']
        std = daily_stats.loc[day, 'std']
        upper_bound = min(mean + std, 1)  # Cap at 1
        lower_bound = max(mean - std, 0) # Cap at 0
        plt.errorbar(day, mean, yerr=[[mean - lower_bound], [upper_bound - mean]], #Asymmetric error
                    fmt='none', ecolor='black', capsize=5)

    # Extended bars to represent "not value"
    plt.bar(daily_stats.index, 1 - daily_stats['mean'], bottom=daily_stats['mean'],
            color='lightblue', label='Non-necrotic')

    plt.xlabel('Day')
    plt.ylabel('Area (%)')
    plt.legend()
    plt.ylim(-0.1, 1.1)  # Ensure the y-axis includes the full range (0 to 1)

def main() -> None:
    #cell_line = "pancreas"
    cell_line = "ht29"
    

    #area_alive, area_dead = dead_alive_area_Pancreas()
    _, area_dead = dead_alive_area_CRC(cell_line)
    
    if cell_line == "ht29":
        area_dead.replace({"day": DICT_DAYS}, inplace=True)
    else:
        area_dead['day'] = area_dead['day']-1
    
    barchart(area_dead)
    
    if cell_line == "ht29":
        plt.title('Progression of HT29 Spheroid Necrosis over Time')
    else:
        plt.title('Progression of Hs-766T Spheroid Necrosis over Time')

    #plt.show()
    plt.savefig(rf"D:\TUW\Tools\calculate_necrotic_area\bar_chart_{cell_line}_no_weighing.jpg")

if __name__ == '__main__':
    main()