// Schedule in home 

#include <预防新型冠状病毒感染的肺炎口罩使用指南>
#include <近期防控新型冠状病毒感染的肺炎工作方案>
... 

extern unsigned long long ENERGY;
extern unsigned long long KNOWLEDGE;

int main( int argc, char*[] argv)
{
    int days_remain = day_of_coronavirus_ends - day_of_coronavirus_begins;
    
    while(days_remain > 0)
    {
        
        int hour = 0, minute = 0;

        __STILL_SLEEPING__
        while(++hour < 7) ;  //zZz...

        wakeup(&hour, &minute), toothBrushing(&hour, &minute);

        __NOT_SLEEPING__

        ENERGY += breakfast(&hour, &minute);

        assert(hour == 8 && minute == 0);
        
        while( ++minute < 30 ) KNOWLEDGE += readingNews();

        ENERGY -= coding(KNOWLEDGE, &hour, &minute);

        if( hour == 12 && isHungry(ENERGY) )
        {
            ENERGY  += lunch(&hour, &minute);
        }
        
        ENERGY -= paperWriting(KNOWLEDGE, &hour, &minute);
        
        if( hour == 16 )
        {
            ENERGY  -= excersice(&hour, &minute);
        }

        if( hour == 18 && isHungry(ENERGY) )
        {
            ENERGY  += dinner(&hour, &minute);
        }

        KNOWLEDGE += reading(&hour, &minute);

        assert(hour >= 21);

        while(++hour < 23) ENERGY -= gaming(&hour, &minute);

        __FALL_IN_SLEEP__

        days_remain --;
    }
    
    int university = ZJU;

    return university;
    
}