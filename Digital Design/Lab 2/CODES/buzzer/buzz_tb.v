`timescale 1ns / 1ps


module buzz_tb;
wire b1, b2;
reg a1, a2, control;
buzzer play(a1, a2, b1, b2, control);
initial
    begin 
        control = 1'b0;
        a1 = 1'b0; a2 = 1'b0;
        #30 a2 = 1'b1;
        #30 a1 = 1'b1;
        #30 a1 = 1'b0; a2 = 1'b0;
        #30 a1 = 1'b1;
        #30 a2 = 1'b1;
        #30 control = 1'b1; a1 = 1'b0; a2 = 1'b0;
        #30 a2 = 1'b1;
        #30 a1 = 1'b1;
        #30 a1 = 1'b0; a2 = 1'b0;
        #30 a1 = 1'b1;
        #30 a2 = 1'b0;
        #30 a1 = 1'b0; a2 = 1'b0;
    end
initial #500 $finish;
endmodule
