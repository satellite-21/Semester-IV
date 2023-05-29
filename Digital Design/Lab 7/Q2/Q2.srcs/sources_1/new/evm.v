`timescale 1ns / 1ps

module evm(A,B,enable,reset,countA,countB);

    input A,B,enable,reset;
    output reg[3:0]countA,countB;
    
    always@(enable,A,B,reset) 
    begin
    
        if(reset) 
            begin 
            countA = 0;
            countB = 0; 
            end
        
        if(enable)
        begin
            if(A==1 && B==0)
            begin
            countA = countA + 1;
            end
            
            else if(A==0 && B==1) 
            begin
            countB = countB + 1; 
            end
            
        end
    end 

endmodule
