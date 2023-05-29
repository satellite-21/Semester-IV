`timescale 1ns / 1ps


module test_bench();

    wire [6:0] d1,d2;
    wire [5:0] out;
    
    reg [2:0] x,y;
    
    multiplier m1(x,y,out,d1,d2);
    
    initial
        begin
        
        x = 3'b000; y = 3'b000;
        
        #10 x = 3'b001; y = 3'b001;
        #10 x = 3'b000; y = 3'b001;
        #10 x = 3'b001; y = 3'b111;
        #10 x = 3'b010; y = 3'b100;
        #10 x = 3'b100; y = 3'b100;
        #10 x = 3'b100; y = 3'b001;
        #10 x = 3'b111; y = 3'b111;
        
        end
    initial #100 $finish;
    
endmodule
