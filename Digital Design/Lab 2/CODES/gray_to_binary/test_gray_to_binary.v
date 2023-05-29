`timescale 1ns / 1ps
module test_gray_to_binary();
wire g1,g2,g3,g4;
reg a,b,c,d;
gray_to_binary gtb1(a,b,c,d,g1,g2,g3,g4);
initial
    begin
        a=1'b0;b=1'b0;c=1'b0;d=1'b0;
        #10
        d=1'b1;
        
        #10
        c=1'b1;
        
        #10
        d=1'b0;
        
        #10
        b=1'b1;
        
        #10
        d=1'b1;
        
        #10
        c=1'b0;
        
        #10
        d=1'b0;
        
        #10
        a=1'b1;
        
        #10
        d=1'b1;
        
        #10
        c=1'b1;
        
        #10
        d=1'b0;
        
        #10
        b=1'b0;
        
        #10
        d=1'b1;
        
        #10
        c=1'b0;
        
        #10
        d=1'b0;
    end
initial #180 $finish;
endmodule
