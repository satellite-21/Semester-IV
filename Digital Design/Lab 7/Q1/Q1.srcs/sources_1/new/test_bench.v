`timescale 1ns / 1ps

module test_shift_and_add_multi;

    reg [3:0] a,b;
    reg clk,load;
    
    wire [7:0] p;
    
    shift_and_add_multi m1(p,a,b,clk,load);
    
    initial
        begin
            
            clk = 0;
            
            a = 4'b0100; b = 4'b0010; load = 1'b0;
            #20 load = 1'b1;
        end
    always #10 clk = ~clk;
    initial #100 $finish; 
    
endmodule
