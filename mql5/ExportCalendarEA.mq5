//+------------------------------------------------------------------+
//| ExportCalendarEA.mq5                                              |
//| Exports MT5 economic calendar to CSV for Python HQTS to read.    |
//| Attach to any chart; runs every 5 min.                           |
//| Output: MQL5/Files/economic_calendar.csv                           |
//+------------------------------------------------------------------+
#property copyright "HQTS"
#property version   "1.00"
#property strict

input int ExportIntervalMinutes = 5;  // Export interval (minutes)
input bool HighImpactOnly = true;     // Export only high-impact events

datetime g_lastExport = 0;
//+------------------------------------------------------------------+
int OnInit()
{
   EventSetTimer(60);  // Check every 60 seconds
   ExportCalendar();
   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
}
//+------------------------------------------------------------------+
void OnTimer()
{
   if(TimeCurrent() - g_lastExport >= ExportIntervalMinutes * 60)
      ExportCalendar();
}
//+------------------------------------------------------------------+
void ExportCalendar()
{
   MqlCalendarValue values[];
   datetime from_time = TimeCurrent();
   datetime to_time = from_time + 86400 * 2;  // Next 2 days

   int n = CalendarValueHistory(values, from_time, to_time, NULL, NULL);
   if(n <= 0)
   {
      Print("ExportCalendarEA: No calendar data (", GetLastError(), ")");
      g_lastExport = TimeCurrent();
      return;
   }

   int handle = FileOpen("economic_calendar.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("ExportCalendarEA: Cannot create file");
      g_lastExport = TimeCurrent();
      return;
   }

   FileWrite(handle, "time_utc", "importance", "currency", "country", "title");
   int count = 0;

   for(int i = 0; i < n; i++)
   {
      MqlCalendarEvent evt;
      if(!CalendarEventById(values[i].event_id, evt))
         continue;

      if(HighImpactOnly && evt.importance != CALENDAR_IMPORTANCE_HIGH)
         continue;

      string imp = "medium";
      if(evt.importance == CALENDAR_IMPORTANCE_HIGH) imp = "high";
      else if(evt.importance == CALENDAR_IMPORTANCE_MODERATE) imp = "medium";
      else if(evt.importance == CALENDAR_IMPORTANCE_LOW) imp = "low";

      long evt_ts = values[i].time;  // Unix timestamp (trade server time)
      FileWrite(handle, IntegerToString(evt_ts), imp, evt.currency, evt.country, evt.name);
      count++;
   }

   FileClose(handle);
   g_lastExport = TimeCurrent();
   Print("ExportCalendarEA: Exported ", count, " events");
}
//+------------------------------------------------------------------+
