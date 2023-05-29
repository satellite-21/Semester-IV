`timescale 1ns / 1ps

module shift_and_add_multi(p,a,b,clk,load);

    output reg[7:0]p;
    input [3:0]a,b;
    input clk,load; 
    
    reg [3:0]x;  
    reg [7:0]y; 
    
    always @(posedge clk)   
    begin
    
        if(load == 1'b0)
            begin 
            x = a;
            y[3:0] = b;
            y[7:4] = 4'b0000;
            end
        
        else
            begin
            if(y[0]) 
            y[7:4] = y[7:4]+ x; 
            y = {1'b0,y[7:1]};
            end

        p=y;   
    end

endmodule