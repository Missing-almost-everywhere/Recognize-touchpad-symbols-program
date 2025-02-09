"""This is an application that identifies which letter a user draws on the touchpad.

The model is trained in a separate file."""

import tkinter as tk
#from tkinter import ttk
import ttkbootstrap as ttk
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import torch
import torch.nn as nn
import math
import os
import pygame
import torchvision.transforms as transforms
import torch.nn.functional as F

# define the tourch model
import torch.nn as nn

input_size=28*28
output_size=62
class NeuralNetwork(nn.Module):
    def __init__(self): # start class
        super().__init__() # call parent init, so nn.Module
        self.flatten = nn.Flatten() # make the input in to a list
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_size)  # Output layer with 62 classes
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model_for_clas= NeuralNetwork()
model_for_clas.load_state_dict(torch.load("model.pth")) if os.path.exists("model.pth") else print("File not found.")

# Set the model to evaluation mode
model_for_clas.eval()

# the model will output value for the neual twork. here i wll defien the functio

# Doing a littel post processing for to get the sumbole for the labesl

labes_from_model=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

test=None

class Touchpad_sumbole_recognizer_app:
    def __init__(self, root):  # Accept root as a parameter
        # Get for the prediction
        self.model=model_for_clas
        self.labes_from_model=labes_from_model

        #For interface
        self.root = root
        self.root.title("Touchpad symbol recognizer")  # Fixed typo "sumbole" -> "symbol"
        self.width=392
        self.top_height=self.width
        self.buttom_widt=50 # number of letters
        self.buttom_height=10 # number of letters
        self.width_line=50
        self.dim_of_picture_in_predciton_model=(28,28)
        self.drawing_tensor = torch.tensor(np.zeros((1, self.dim_of_picture_in_predciton_model[0], self.dim_of_picture_in_predciton_model[1]), dtype=np.float32)) # picture drawn with based on events
        self.image_tensor = torch.tensor(np.zeros((1, self.dim_of_picture_in_predciton_model[0], self.dim_of_picture_in_predciton_model[1]), dtype=np.float32)) # picture based on screenshot from the screen
        
        # For brush 
        self.size_pensel=20
        self.image_tensor_brush=  torch.tensor(np.zeros((1, self.width+2*self.size_pensel,self.width+2*self.size_pensel), dtype=np.float32))

        # Set up the frame for layout management
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Set up top canvas for drawing
        self.canvas_top = tk.Canvas(self.frame, bg="white", width=self.width, height=self.top_height)
        self.canvas_top.grid(row=0, column=0, padx=0, pady=0)  # Use grid() for layout
        

        # Bind events to the top canvas
        self.canvas_top.bind("<B1-Motion>", self.brush)  # Drag to draw
        self.canvas_top.bind("<Button-3>", self.clear_canvas)  # Right-click to clear

        # Store last touch/mouse position
        self.last_x, self.last_y = None, None

        # middel her is buttom to summit letter 
        self.button_middel = tk.Button(self.frame, text="Summit letter",width=50, command=self.summit_sumbole)
        self.button_middel.grid(row=1,column=0,padx=0,pady=0)

        # Set up bottom canvas (e.g., for displaying the result)
        self.text_bottom = tk.Text(self.frame, bg="white", width=self.buttom_widt, height=self.buttom_height)
        self.text_bottom.grid(row=2, column=0, padx=0, pady=0)  # Use grid() for layout
    # make drawing in the top window
    def draw(self, event):
        """Here the point is drawn"""
        if self.last_x and self.last_y:
            self.canvas_top.create_line(self.last_x, self.last_y, event.x, event.y, fill="black", width=20, capstyle=tk.ROUND)
        # Update last position
        self.last_x, self.last_y = event.x, event.y
        # event.x here is condinates from top left.
        """"The x and y position need to put in the tensor, but their is size differnce so a transformation is used"""
        x_postion =min(self.last_x//(self.width/self.dim_of_picture_in_predciton_model[0]),self.dim_of_picture_in_predciton_model[0]-1) # the min is to ensure the last pix don go out bounds for the index
        # the tensor is index from buttom left, so i need a different transformation
        y_postion =self.dim_of_picture_in_predciton_model[1]-1-min(self.dim_of_picture_in_predciton_model[1]- self.last_y//(self.top_height/self.dim_of_picture_in_predciton_model[1]),self.dim_of_picture_in_predciton_model[1]-1)
        self.drawing_tensor[0, int(y_postion), int(x_postion)] = 1.# had to change x and y postion since row in tensor is colums

    def brush(self,event):
        """Brush is extension of draw where i their is cirkel """
        if self.last_x and self.last_y:
            self.canvas_top.create_line(self.last_x, self.last_y, event.x, event.y, fill="black", width=20, capstyle=tk.ROUND)
        # Update last position
        self.last_x, self.last_y = event.x, event.y
        # event.x here is condinates from top left.
        """"The x and y position need to put in the tensor, but their is size differnce so a transformation is used"""
        x_postion =self.size_pensel+self.last_x
        y_postion =self.size_pensel+self.last_y
        self.image_tensor_brush[0, int(y_postion), int(x_postion)] = 1.
        # The last 
        # divid it to section.
        n_ind=self.size_pensel

        for j in range(self.size_pensel):
            for i in range(self.size_pensel):
                self.image_tensor_brush[0, int(y_postion)+i, int(x_postion)+j] =1.
                self.image_tensor_brush[0, int(y_postion)-i, int(x_postion)-j] =1.
                self.image_tensor_brush[0, int(y_postion)-i, int(x_postion)+j] =1.
                self.image_tensor_brush[0, int(y_postion)+i, int(x_postion)-j] =1.
                n_ind=n_ind-1
        
        
    def get_screen_shoot_convetered_to_tensor(self):
        self.root.update_idletasks() # Ensure all positions are updated
        self.canvas_top.update_idletasks()
        x = self.root.winfo_rootx()+self.canvas_top.winfo_x()
        y = self.root.winfo_rooty()+self.canvas_top.winfo_y()#self.canvas_top.winfo_y() +self.root.winfo_rooty()
        x1 = x + self.width
        y1 = y + self.top_height
        # Capture the canvas as an image
        image = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28 if needed
        # Convert to tensor
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(1, 28, 28) / 255.0

        print(tensor.shape)  # Output should be (1, 28, 28)
        return tensor
        
    def clear_canvas(self):
        """Clear the canvas"""
        self.drawing_tensor = torch.tensor(np.zeros((1, 28, 28), dtype=np.float32))
        self.image_tensor_brush=  torch.tensor(np.zeros((1, self.width+2*self.size_pensel,self.width+2*self.size_pensel), dtype=np.float32))
        self.image_tensor = torch.tensor(np.zeros((1, self.dim_of_picture_in_predciton_model[0], self.dim_of_picture_in_predciton_model[1]), dtype=np.float32)) # picture based on screenshot from the screen

        self.canvas_top.delete("all")

    def reset_position(self, event):
        """Reset the drawing position when the mouse is released"""
        self.last_x, self.last_y = None, None
    def get_sumbole(self,Img_tensor):  # Accept event as a parameter
        """This function make return a label for the predctio model"""
        self.model.eval()
        output_val=self.model(Img_tensor)
        probabilities = torch.softmax(output_val, dim=1)
        predict_label_index=torch.argmax(probabilities, dim=1)
        predicted_label = self.labes_from_model[predict_label_index]
        return predicted_label
    
    def add_letter(self,letter):
        """Add a letter to the text section of the aplication"""
        string_for_display = self.text_bottom.get("1.0", tk.END).strip() + str(letter)
        self.text_bottom.delete("1.0", tk.END) # remove the stirng
        # sumit the stirng for display
        self.text_bottom.insert(tk.END, string_for_display)
    
    def summit_sumbole(self):
        """This function connet the draw picture and the add_letter function"""
        resized_tensor = F.interpolate(self.image_tensor_brush[:,self.size_pensel:-self.size_pensel,self.size_pensel:-self.size_pensel].unsqueeze(0), size=self.dim_of_picture_in_predciton_model, mode='bilinear', align_corners=False)
        sumbole=self.get_sumbole(resized_tensor)
        self.add_letter(sumbole)
        self.clear_canvas()
        


    
    


# run the aplication

if __name__ == "__main__":
    root = tk.Tk()
    app = Touchpad_sumbole_recognizer_app(root)  # Pass root to the app constructor

    # Reset drawing position when releasing the mouse/touch
    root.bind("<ButtonRelease-1>", app.reset_position)

    root.mainloop()

