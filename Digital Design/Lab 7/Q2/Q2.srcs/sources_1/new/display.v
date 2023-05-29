`timescale 1ns / 1ps

module disp_result(A_result,B_result,A_display,B_display,display_result);

   input[0:3]A_result,B_result;
   output reg[0:6]A_display,B_display;
   input display_result;

   always@(display_result)
   if(display_result) begin
     case(A_result)
       0 : A_display = 7'b1111110;
       1 : A_display = 7'b0110000;
       2 : A_display = 7'b1101101;
       3 : A_display = 7'b1111001;
       4 : A_display = 7'b0110011;
       5 : A_display = 7'b1011011;
       6 : A_display = 7'b1011111;
       7 : A_display = 7'b1110000;
       8 : A_display = 7'b1111111;
       9 : A_display = 7'b1111011;
      default : A_display=7'b0000000;
     endcase
     
     case(B_result)
       0 : B_display = 7'b1111110;
       1 : B_display = 7'b0110000;
       2 : B_display = 7'b1101101;
       3 : B_display = 7'b1111001;
       4 : B_display = 7'b0110011;
       5 : B_display = 7'b1011011;
       6 : B_display = 7'b1011111;
       7 : B_display = 7'b1110000;
       8 : B_display = 7'b1111111;
       9 : B_display = 7'b1111011;
      default : B_display=7'b0000000;
     endcase    
   end
   else begin
     B_display = 7'b1111110;
     A_display = 7'b1111110;
     end
endmodule
